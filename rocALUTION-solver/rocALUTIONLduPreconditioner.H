/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#ifndef rocALUTIONLduPreconditioner_H
#define rocALUTIONLduPreconditioner_H

#include "lduMatrix.H"

#include <string>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
             Class rocALUTIONLduPreconditioner Declaration
\*---------------------------------------------------------------------------*/

//- GPU algebraic-multigrid preconditioner (one AMG V-cycle per call).
//
//  Unlike rocALUTIONLduSolver this object does not solve the system: it
//  applies M^-1 to the residual inside OpenFOAM's own Krylov solver (PCG,
//  PBiCGStab, ...). That is what makes it correct in parallel: OpenFOAM keeps
//  the global iterations, the reductions and the processor-interface
//  exchange, and the GPU only handles the local block (additive Schwarz).
class rocALUTIONLduPreconditioner
:
    public lduMatrix::preconditioner
{
    // Private Data

        //- Cache key (mesh pointer + field name)
        std::string key_;

        //- Configuration captured from the solver dictionary
        int device_;
        int nSweeps_;
        bool timing_;


public:

    //- Runtime type information
    TypeName("rocALUTION");


    // Constructors

        rocALUTIONLduPreconditioner
        (
            const lduMatrix::solver& sol,
            const dictionary& solverControls
        );


    //- Destructor
    virtual ~rocALUTIONLduPreconditioner() = default;


    // Member Functions

        virtual void precondition
        (
            solveScalarField& wA,
            const solveScalarField& rA,
            const direction cmpt = 0
        ) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
