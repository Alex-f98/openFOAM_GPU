/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#include "rocALUTIONLduSolver.H"
#include "rocALUTIONCommon.H"
#include "PrecisionAdaptor.H"

#include <algorithm>
#include <cmath>
#include <fstream>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

using namespace rocALUTIONFoam;


// * * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * //

defineTypeNameAndDebug(rocALUTIONLduSolver, 0);

lduMatrix::solver::addsymMatrixConstructorToTable<rocALUTIONLduSolver>
    addrocALUTIONLduSolverSymMatrixConstructorToTable_;

lduMatrix::solver::addasymMatrixConstructorToTable<rocALUTIONLduSolver>
    addrocALUTIONLduSolverAsymMatrixConstructorToTable_;


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::rocALUTIONLduSolver::rocALUTIONLduSolver
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::solverPerformance Foam::rocALUTIONLduSolver::scalarSolve
(
    solveScalarField& psi,
    const solveScalarField& source,
    const direction cmpt
) const
{
    solverPerformance solverPerf(typeName, fieldName_);

    const label nCells = psi.size();
    if (nCells == 0)
    {
        return solverPerf;
    }

    // --- OpenFOAM residual bookkeeping (same convention as PCG) ---
    solveScalarField wA(nCells);
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    solveScalarField rA(source - wA);
    solveScalarField pA(nCells);

    const solveScalar normFactor =
        this->normFactor(psi, source, wA, pA);

    solverPerf.initialResidual() =
        gSumMag(rA, matrix().mesh().comm())/normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    if
    (
        minIter_ == 0
     && solverPerf.checkConvergence(tolerance_, relTol_, log_)
    )
    {
        return solverPerf;
    }

    // --- Cache lookup (state survives the solver object) ---
    const std::string key =
        std::to_string
        (
            reinterpret_cast<std::uintptr_t>(&matrix_.mesh())
        )
      + "/" + fieldName_;

    auto& reg = registry();
    auto it = reg.find(key);
    if (it == reg.end())
    {
        it = reg.emplace(key, std::unique_ptr<RocEntry>(new RocEntry())).first;
    }
    RocEntry& e = *it->second;

    const std::string precond =
        controlDict_.getOrDefault<word>("rocPreconditioner", "AMG");
    const int device = controlDict_.getOrDefault<int>("rocDevice", 0);
    const bool timing = controlDict_.getOrDefault<bool>("rocTiming", false);

    ensureRocALUTION(device);

    double tFill = 0.0;
    double tUpdate = 0.0;
    double tTransfer = 0.0;
    double tSolve = 0.0;

    // --- (Re)build structure if topology or solver config changed ---
    if
    (
        !e.built
     || e.addressing != &matrix_.lduAddr()
     || e.nCells != nCells
     || e.precond != precond
    )
    {
        const auto tBuildStart = Clock::now();

        // Drop previous objects before re-creating
        e.solver = nullptr;
        e.cg.reset();
        e.bicg.reset();
        e.amg.reset();
        e.saamg.reset();
        e.jac.reset();
        e.x.reset();
        e.b.reset();
        e.A.reset();

        buildPattern(e, matrix_);

        //- Only the negative-definite matrices (pressure) need negating
        e.negate = (gAverage(matrix_.diag()) < 0);

        const int nnz = e.rowPtr.back();

        int* rp = new int[e.nCells + 1];
        int* ci = new int[nnz];
        double* vv = new double[nnz];

        std::copy(e.rowPtr.begin(), e.rowPtr.end(), rp);
        std::copy(e.colIdx.begin(), e.colIdx.end(), ci);
        {
            std::vector<double> tmp(nnz);
            e.coefSum = fillValues(e, matrix_, tmp);
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

        if (matrix_.symmetric())
        {
            e.cg.reset(new rocalution::CG<RocMat, RocVec, double>());
            e.cg->Verbose(0);
            e.solver = e.cg.get();
        }
        else
        {
            e.bicg.reset(new rocalution::BiCGStab<RocMat, RocVec, double>());
            e.bicg->Verbose(0);
            e.solver = e.bicg.get();
        }

        e.solver->SetOperator(*e.A);

        if (precond == "Jacobi")
        {
            e.jac.reset(new rocalution::Jacobi<RocMat, RocVec, double>());
            e.solver->SetPreconditioner(*e.jac);
        }
        else if (precond == "SAAMG")
        {
            e.saamg.reset(new rocalution::SAAMG<RocMat, RocVec, double>());
            e.saamg->Verbose(0);
            e.solver->SetPreconditioner(*e.saamg);
        }
        else if (precond != "none")
        {
            e.amg.reset
            (
                new rocalution::RugeStuebenAMG<RocMat, RocVec, double>()
            );
            e.amg->Verbose(0);
            e.solver->SetPreconditioner(*e.amg);
        }

        e.solver->SetResidualNorm(2);
        e.solver->InitMaxIter(maxIter_);
        e.solver->Build();

        e.precond = precond;
        e.built = true;
        e.buildMs = msSince(tBuildStart);

        if (log_ >= 1)
        {
            const scalarField& d = matrix_.diag();
            const scalarField& up = matrix_.upper();

            Info<< "rocALUTION build: field=" << fieldName_
                << " nCells=" << e.nCells
                << " nnz=" << nnz
                << " symmetric=" << matrix_.symmetric()
                << " precon=" << precond
                << " negate=" << e.negate
                << " diag=[" << min(d) << ", " << max(d) << "]"
                << " upper=[" << min(up) << ", " << max(up) << "]"
                << " time=" << e.buildMs << " ms"
                << endl;
        }
    }
    else
    {
        // --- Numeric update: pattern frozen, only coefficients change ---
        const int nnz = e.rowPtr.back();

        auto t0 = Clock::now();

        std::vector<double> val(nnz);
        const double cs = fillValues(e, matrix_, val);

        tFill = msSince(t0);

        // The two pressure solves of a PISO corrector share the same
        // matrix, as do the three momentum components: skip the (expensive)
        // numeric rebuild when the coefficients are bit-identical.
        if (cs != e.coefSum)
        {
            t0 = Clock::now();

            e.A->UpdateValuesCSR(val.data());
            e.solver->ReBuildNumeric();

            tUpdate = msSince(t0);
            e.coefSum = cs;
        }
    }

    // --- Map OpenFOAM's convergence criterion to rocALUTION's ---
    //
    // rocALUTION's relative criterion is measured against the starting
    // residual r_1 = b - A*x_init (see IterativeLinearSolver), while
    // OpenFOAM accepts
    //     R_final <= max(tolerance, relTol*R_initial)
    // Both are ratios of the same residual, hence
    //     rocALUTION rel_tol = target/initialResidual
    // gives the same convergence level.
    const solveScalar target =
        std::max(tolerance_, relTol_*solverPerf.initialResidual());

    //  A safety factor compensates the difference between OpenFOAM's L1
    //  normalisation and rocALUTION's L2 residual (typically 2-3x).
    const double safety = 4.0;

    const double tolRel = std::min
    (
        1.0,
        std::max
        (
            double
            (
                target
               /(safety*std::max(solverPerf.initialResidual(), 1e-300))
            ),
            1e-14
        )
    );

    // ||b||, reported for diagnostics only
    double b2 = 0.0;
    for (label i = 0; i < nCells; ++i)
    {
        b2 += double(source[i])*double(source[i]);
    }
    b2 = returnReduce(b2, sumOp<double>());
    const double bnorm = std::sqrt(b2);

    // --- Upload rhs / initial guess ---
    std::vector<double> bh(nCells);
    std::vector<double> xh(nCells);
    for (label i = 0; i < nCells; ++i)
    {
        bh[i] = (e.negate ? -1.0 : 1.0)*double(source[i]);
        xh[i] = double(psi[i]);
    }

    // NOTE: CopyFromData()/CopyToData() only operate on host-resident
    // vectors. Calling them on an accelerated vector silently leaves the
    // device buffer untouched, so the solver would run on stale data.
    auto t0 = Clock::now();

    e.b->MoveToHost();
    e.b->CopyFromData(bh.data());
    e.b->MoveToAccelerator();

    e.x->MoveToHost();
    e.x->CopyFromData(xh.data());
    e.x->MoveToAccelerator();

    tTransfer = msSince(t0);

    // --- Optional one-shot export of the local system (offline kernels) ---
    if
    (
        controlDict_.getOrDefault<bool>("rocDump", false)
     && fieldName_ == "p"
    )
    {
        static bool dumped = false;
        if (!dumped)
        {
            std::vector<double> vv(e.rowPtr.back());
            fillValues(e, matrix_, vv);

            std::ofstream os("roc_matrix_p.bin", std::ios::binary);
            const std::int64_t n = e.nCells;
            const std::int64_t nnz = e.rowPtr.back();
            os.write(reinterpret_cast<const char*>(&n), sizeof(n));
            os.write(reinterpret_cast<const char*>(&nnz), sizeof(nnz));
            os.write
            (
                reinterpret_cast<const char*>(e.rowPtr.data()),
                sizeof(int)*(n + 1)
            );
            os.write
            (
                reinterpret_cast<const char*>(e.colIdx.data()),
                sizeof(int)*nnz
            );
            os.write
            (
                reinterpret_cast<const char*>(vv.data()),
                sizeof(double)*nnz
            );
            os.write
            (
                reinterpret_cast<const char*>(bh.data()),
                sizeof(double)*n
            );
            dumped = true;

            Info<< "rocALUTION: wrote roc_matrix_p.bin (n=" << n
                << ", nnz=" << nnz << ')' << endl;
        }
    }

    // --- Solve ---
    t0 = Clock::now();

    e.solver->InitMaxIter(maxIter_);
    e.solver->InitMinIter(minIter_);
    e.solver->InitTol(0.0, tolRel, 1e8);
    e.solver->Solve(*e.b, e.x.get());

    tSolve = msSince(t0);

    t0 = Clock::now();

    e.x->MoveToHost();
    e.x->CopyToData(xh.data());
    e.x->MoveToAccelerator();

    tTransfer += msSince(t0);

    for (label i = 0; i < nCells; ++i)
    {
        psi[i] = solveScalar(xh[i]);
    }

    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
    rA = source - wA;

    solverPerf.finalResidual() =
        gSumMag(rA, matrix().mesh().comm())/normFactor;

    solverPerf.nIterations() = e.solver->GetIterationCount();

    if (timing)
    {
        Info<< "rocALUTION timing: field=" << fieldName_
            << " fill=" << tFill
            << " update+rebuild=" << tUpdate
            << " transfer=" << tTransfer
            << " solve=" << tSolve
            << " iters=" << solverPerf.nIterations()
            << " target=" << target
            << " tolRel=" << tolRel
            << " normFactor=" << normFactor
            << " bnorm=" << bnorm
            << " status=" << e.solver->GetSolverStatus()
            << " coefSum=" << e.coefSum
            << endl;
    }

    if (log_ >= 2)
    {
        //- Verify that the CSR operator matches OpenFOAM's Amul
        std::vector<double> vv(e.rowPtr.back());
        fillValues(e, matrix_, vv);

        solveScalarField wA2(nCells);
        matrix_.Amul(wA2, psi, interfaceBouCoeffs_, interfaces_, cmpt);

        const double sgn = (e.negate ? -1.0 : 1.0);

        double maxd = 0.0;
        double d1 = 0.0;
        for (label i = 0; i < nCells; ++i)
        {
            double yvi = 0.0;
            for (int p = e.rowPtr[i]; p < e.rowPtr[i + 1]; ++p)
            {
                yvi += vv[p]*double(psi[e.colIdx[p]]);
            }
            const double d = std::abs(yvi - sgn*double(wA2[i]));
            maxd = std::max(maxd, d);
            d1 += d;
        }

        Info<< "rocALUTION check: max|A_csr*x - A_of*x| = " << maxd
            << "  L1=" << d1 << endl;
    }

    return solverPerf;
}


Foam::solverPerformance Foam::rocALUTIONLduSolver::solve
(
    scalarField& psi_s,
    const scalarField& source,
    const direction cmpt
) const
{
    PrecisionAdaptor<solveScalar, scalar> tpsi(psi_s);

    return scalarSolve
    (
        tpsi.ref(),
        ConstPrecisionAdaptor<solveScalar, scalar>(source)(),
        cmpt
    );
}


} // End namespace Foam


// ************************************************************************* //
