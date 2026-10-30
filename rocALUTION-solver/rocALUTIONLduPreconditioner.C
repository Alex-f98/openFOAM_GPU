/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#include "rocALUTIONLduPreconditioner.H"
#include "rocALUTIONCommon.H"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <vector>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

using namespace rocALUTIONFoam;

// * * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * //

defineTypeNameAndDebug(rocALUTIONLduPreconditioner, 0);

lduMatrix::preconditioner::addsymMatrixConstructorToTable
<
    rocALUTIONLduPreconditioner
>
    addrocALUTIONLduPreconditionerSymMatrixConstructorToTable_;

lduMatrix::preconditioner::addasymMatrixConstructorToTable
<
    rocALUTIONLduPreconditioner
>
    addrocALUTIONLduPreconditionerAsymMatrixConstructorToTable_;


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::rocALUTIONLduPreconditioner::rocALUTIONLduPreconditioner
(
    const lduMatrix::solver& sol,
    const dictionary& solverControls
)
:
    lduMatrix::preconditioner(sol),
    key_
    (
        "precond/"
      + std::to_string
        (
            reinterpret_cast<std::uintptr_t>(&sol.matrix().mesh())
        )
      + "/" + sol.fieldName()
    ),
    device_(solverControls.getOrDefault<int>("rocDevice", 0)),
    nSweeps_(solverControls.getOrDefault<int>("rocAMGSweeps", 1)),
    timing_(solverControls.getOrDefault<bool>("rocTiming", false))
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::rocALUTIONLduPreconditioner::precondition
(
    solveScalarField& wA,
    const solveScalarField& rA,
    const direction
) const
{
    const lduMatrix& m = solver_.matrix();
    const label nCells = wA.size();

    if (!nCells)
    {
        return;
    }

    auto& reg = precondRegistry();
    auto it = reg.find(key_);
    if (it == reg.end())
    {
        it = reg.emplace(key_, std::unique_ptr<RocEntry>(new RocEntry())).first;
    }
    RocEntry& e = *it->second;

    ensureRocALUTION(device_);

    // --- (Re)build the hierarchy when the mesh or the pattern changes ---
    if
    (
        !e.built
     || e.addressing != &m.lduAddr()
     || e.nCells != nCells
    )
    {
        const auto tBuild = Clock::now();

        e.solver = nullptr;
        e.amg.reset();
        e.saamg.reset();
        e.jac.reset();
        e.x.reset();
        e.b.reset();
        e.A.reset();

        buildPattern(e, m);

        //- Only the negative-definite matrices (pressure) need negating
        e.negate = (gAverage(m.diag()) < 0);

        const int nnz = e.rowPtr.back();

        int* rp = new int[e.nCells + 1];
        int* ci = new int[nnz];
        double* vv = new double[nnz];

        std::copy(e.rowPtr.begin(), e.rowPtr.end(), rp);
        std::copy(e.colIdx.begin(), e.colIdx.end(), ci);
        {
            std::vector<double> tmp(nnz);
            e.coefSum = fillValues(e, m, tmp);
            std::copy(tmp.begin(), tmp.end(), vv);
        }

        e.A.reset(new RocMat());
        // rocALUTION takes ownership of rp/ci/vv
        e.A->SetDataPtrCSR(&rp, &ci, &vv, "A", nnz, e.nCells, e.nCells);
        e.A->MoveToAccelerator();

        e.x.reset(new RocVec());
        e.b.reset(new RocVec());
        e.x->Allocate("x", e.nCells);
        e.b->Allocate("b", e.nCells);
        e.x->MoveToAccelerator();
        e.b->MoveToAccelerator();

        e.amg.reset(new rocalution::RugeStuebenAMG<RocMat, RocVec, double>());
        e.amg->Verbose(0);
        e.amg->SetOperator(*e.A);
        e.amg->SetResidualNorm(2);
        e.amg->InitMaxIter(nSweeps_);
        e.amg->InitMinIter(0);
        e.amg->Build();

        e.built = true;
        e.buildMs = msSince(tBuild);

        if (Pstream::master())
        {
            Info<< "rocALUTION preconditioner: field=" << solver_.fieldName()
                << " nCells=" << e.nCells
                << " nnz=" << nnz
                << " symmetric=" << m.symmetric()
                << " negate=" << e.negate
                << " procs=" << Pstream::nProcs()
                << " time=" << e.buildMs << " ms"
                << endl;
        }
    }
    else
    {
        const int nnz = e.rowPtr.back();
        std::vector<double> val(nnz);
        const double cs = fillValues(e, m, val);

        if (cs != e.coefSum)
        {
            e.A->UpdateValuesCSR(val.data());
            e.amg->ReBuildNumeric();
            e.coefSum = cs;
        }
    }

    // --- Apply one V-cycle to the local residual: x = 0, b = sgn*r ---
    const auto tStart = Clock::now();

    std::vector<double> bh(nCells);
    const double sgn = (e.negate ? -1.0 : 1.0);
    for (label i = 0; i < nCells; ++i)
    {
        bh[i] = sgn*double(rA[i]);
    }

    e.b->MoveToHost();
    e.b->CopyFromData(bh.data());
    e.b->MoveToAccelerator();

    e.x->Zeros();

    e.amg->InitMaxIter(nSweeps_);
    e.amg->InitMinIter(0);
    e.amg->InitTol(0.0, 0.0, 1e9);
    e.amg->Solve(*e.b, e.x.get());

    e.x->MoveToHost();
    e.x->CopyToData(bh.data());
    e.x->MoveToAccelerator();

    for (label i = 0; i < nCells; ++i)
    {
        wA[i] = solveScalar(bh[i]);
    }

    if (timing_)
    {
        Info<< "rocALUTION precond timing: field=" << solver_.fieldName()
            << " proc=" << Pstream::myProcNo()
            << " apply=" << msSince(tStart) << " ms"
            << endl;
    }
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
