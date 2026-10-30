/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#ifndef rocALUTIONCommon_H
#define rocALUTIONCommon_H

#include "lduMatrix.H"

#include <rocalution/rocalution.hpp>

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace rocALUTIONFoam
{

typedef rocalution::LocalMatrix<double> RocMat;
typedef rocalution::LocalVector<double> RocVec;
typedef rocalution::IterativeLinearSolver<RocMat, RocVec, double> RocSolver;

typedef std::chrono::steady_clock Clock;

inline double msSince(const Clock::time_point& start)
{
    return std::chrono::duration<double, std::milli>
    (
        Clock::now() - start
    ).count();
}


//- Persistent state per (mesh, field).
//
//  OpenFOAM creates and destroys the solver and the preconditioner objects on
//  every solve() call, so nothing can be cached in them: the CSR pattern and
//  the AMG hierarchy live here.
//
//  In parallel each rank keeps its own local block (internal faces only);
//  interface coefficients live in OpenFOAM and are applied by Amul/interfaces.
struct RocEntry
{
    const lduAddressing* addressing = nullptr;
    label nCells = 0;

    // CSR pattern (host mirror of the structure handed to rocALUTION)
    std::vector<int> rowPtr;
    std::vector<int> colIdx;
    std::vector<int> entryFace;    // -1 -> diagonal
    std::vector<char> entryUpper;  // 1 -> upper[face], 0 -> lower[face]

    // rocALUTION objects
    std::unique_ptr<RocMat> A;
    std::unique_ptr<RocVec> x;
    std::unique_ptr<RocVec> b;
    std::unique_ptr<rocalution::CG<RocMat, RocVec, double>> cg;
    std::unique_ptr<rocalution::BiCGStab<RocMat, RocVec, double>> bicg;
    std::unique_ptr<rocalution::RugeStuebenAMG<RocMat, RocVec, double>> amg;
    std::unique_ptr<rocalution::SAAMG<RocMat, RocVec, double>> saamg;
    std::unique_ptr<rocalution::Jacobi<RocMat, RocVec, double>> jac;
    RocSolver* solver = nullptr;

    bool built = false;
    bool negate = true;
    std::string precond;

    // Fingerprint of the coefficients currently uploaded to rocALUTION
    double coefSum = -1.0;

    double buildMs = 0.0;
};


//- Two registries: the solver and the preconditioner never share state.
inline std::map<std::string, std::unique_ptr<RocEntry>>& registry()
{
    // Intentionally leaked: avoids HIP destruction-order problems at exit
    static auto* reg = new std::map<std::string, std::unique_ptr<RocEntry>>();
    return *reg;
}

inline std::map<std::string, std::unique_ptr<RocEntry>>& precondRegistry()
{
    static auto* reg = new std::map<std::string, std::unique_ptr<RocEntry>>();
    return *reg;
}


inline void ensureRocALUTION(const int device)
{
    static bool initialised = false;

    if (!initialised)
    {
        rocalution::set_device_rocalution(device);
        rocalution::init_rocalution();

        initialised = true;
    }
}


//- Build the CSR pattern from the LDU addressing (topology only)
inline void buildPattern(RocEntry& e, const lduMatrix& m)
{
    const labelUList& u = m.lduAddr().upperAddr();
    const labelUList& l = m.lduAddr().lowerAddr();

    const label n = m.diag().size();
    const label nF = u.size();

    e.addressing = &m.lduAddr();
    e.nCells = n;

    std::vector<int> deg(n, 1);   // diagonal
    for (label f = 0; f < nF; ++f)
    {
        ++deg[u[f]];
        ++deg[l[f]];
    }

    e.rowPtr.resize(n + 1);
    e.rowPtr[0] = 0;
    for (label i = 0; i < n; ++i)
    {
        e.rowPtr[i + 1] = e.rowPtr[i] + deg[i];
    }

    const int nnz = e.rowPtr[n];
    e.colIdx.assign(nnz, 0);
    e.entryFace.assign(nnz, -1);
    e.entryUpper.assign(nnz, 0);

    std::vector<int> cur(e.rowPtr.begin(), e.rowPtr.end() - 1);

    // Diagonal
    for (label i = 0; i < n; ++i)
    {
        e.colIdx[cur[i]++] = int(i);
    }

    // Off-diagonals (matching lduMatrix::Amul):
    //     A(upperAddr[f], lowerAddr[f]) = lower[f]
    //     A(lowerAddr[f], upperAddr[f]) = upper[f]
    // This distinction only matters for asymmetric matrices (e.g. momentum);
    // for the symmetric pressure matrix upper == lower.
    for (label f = 0; f < nF; ++f)
    {
        const int pu = cur[u[f]]++;
        e.colIdx[pu] = int(l[f]);
        e.entryFace[pu] = int(f);
        e.entryUpper[pu] = 0;

        const int pl = cur[l[f]]++;
        e.colIdx[pl] = int(u[f]);
        e.entryFace[pl] = int(f);
        e.entryUpper[pl] = 1;
    }

    // Sort each row by column index (rows have ~7 entries: insertion sort)
    for (label i = 0; i < n; ++i)
    {
        for (int p = e.rowPtr[i] + 1; p < e.rowPtr[i + 1]; ++p)
        {
            const int c = e.colIdx[p];
            const int f = e.entryFace[p];
            const char up = e.entryUpper[p];

            int q = p - 1;
            while (q >= e.rowPtr[i] && e.colIdx[q] > c)
            {
                e.colIdx[q + 1] = e.colIdx[q];
                e.entryFace[q + 1] = e.entryFace[q];
                e.entryUpper[q + 1] = e.entryUpper[q];
                --q;
            }
            e.colIdx[q + 1] = c;
            e.entryFace[q + 1] = f;
            e.entryUpper[q + 1] = up;
        }
    }
}


//- Fill the CSR values from the current LDU coefficients.
//
//  rocALUTION expects an M-matrix (positive diagonal, negative
//  off-diagonals). OpenFOAM's pressure matrix has the opposite sign
//  convention while the momentum matrix already matches, so the sign is
//  chosen per field (`e.negate`).
//- Returns the sum of the written values (a cheap fingerprint used to
//  detect that the matrix is unchanged between two solves)
inline double fillValues
(
    const RocEntry& e,
    const lduMatrix& m,
    std::vector<double>& val
)
{
    const scalarField& d = m.diag();
    const scalarField& up = m.upper();
    const scalarField& lo = m.lower();

    const double sgn = (e.negate ? -1.0 : 1.0);

    double sum = 0.0;

    for (label i = 0; i < e.nCells; ++i)
    {
        for (int p = e.rowPtr[i]; p < e.rowPtr[i + 1]; ++p)
        {
            const int f = e.entryFace[p];
            val[p] = sgn*((f < 0) ? double(d[i])
                                  : (e.entryUpper[p] ? double(up[f])
                                                     : double(lo[f])));
            sum += val[p];
        }
    }

    return sum;
}

} // End namespace rocALUTIONFoam
} // End namespace Foam

#endif

// ************************************************************************* //
