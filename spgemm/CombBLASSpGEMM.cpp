


#include <mpi.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cassert>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/ParFriends.h"
//#include "CombBLAS/ParFriendsExt.h"
//#include "gspgemm/meSpGEMM.h"

#define VERBOSE

#ifdef VERBOSE
#define PRINT(msg) std::cout<<msg<<std::endl;
#else
#define PRINT(msg)
#endif

#define ITERS 4 //First one doesn't impact the average

#define TIMING
#define THREADED

#ifdef TIMING
double cblas_alltoalltime;
double cblas_allgathertime;
//////////////////////////
double mcl_Abcasttime;
double mcl_Bbcasttime;
double mcl_localspgemmtime;
double mcl_multiwaymergetime;
double mcl_kselecttime;
double mcl_prunecolumntime;
double mcl_symbolictime;
double mcl_totaltime;
double mcl_tt;
int64_t mcl_nnzc;
///////////////////////////
double mcl_Abcasttime_prev;
double mcl_Bbcasttime_prev;
double mcl_localspgemmtime_prev;
double mcl_multiwaymergetime_prev;
double mcl_kselecttime_prev;
double mcl_prunecolumntime_prev;
double mcl_symbolictime_prev;
double mcl_totaltime_prev;
double mcl_tt_prev;
int64_t mcl_nnzc_prev;
///////////////////////////
 double mcl3d_conversiontime;
 double mcl3d_symbolictime;
 double mcl3d_Abcasttime;
 double mcl3d_Bbcasttime;
 double mcl3d_SUMMAtime;
 double mcl3d_localspgemmtime;
 double mcl3d_SUMMAmergetime;
 double mcl3d_reductiontime;
 double mcl3d_3dmergetime;
 double mcl3d_kselecttime;

#endif


using namespace combblas;


void runSpGEMM1Dcpu(int argc, char ** argv) {
    //TODO
}


enum SPGEMMALG2D {
    MEM_EFFICIENT_2D_GPU,
    MEM_EFFICIENT_2D,
    DOUBLE_BUFF,
    SYNCH,
    OVERLAP,
    BLOCKED
} typedef SPGEMMALG2D;

/* 
 * Run a SpGEMM2D algorithm.
 * The value of the 2nd command line arg determines which algorithm is run
 * 0: MemEfficientSpGEMMg
 * 1: MemEfficientSpGEMM
 * 2: Mult_AnXBn_DoubleBuff
 * 3: Mult_AnXBn_Synch
 * 4: blocked spgemm
 */

void runSpGEMM2Dcpu(int argc, char ** argv) {

    int np; int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    /* Setup comm grid and sparse matrix */
    std::string matName(argv[2]);
    if (rank==0)
        PRINT("Matrix name: "+matName+"\n");
    
    std::shared_ptr<CommGrid> grid;
    grid.reset(new CommGrid(MPI_COMM_WORLD,0,0));

    /* Useful types */
    typedef double NT;
    typedef int64_t IT;
    typedef SpDCCols <IT, NT> DER;
    
    SpParMat<IT, NT, DER> A(grid);
    
    /* Read in matrix */
    A.ParallelReadMM(matName, true, maximum<NT>());
    
    double loadBalance = A.LoadImbalance();
    if (rank==0)
        PRINT("Load balance: " + std::to_string(loadBalance) + "\n");

    /* Matrix B, randomly permute columns */
    SpParMat<IT, NT, DER> B(A);
    FullyDistVec<IT, NT> p(A.getcommgrid());
    p.iota(A.getnrow(), 0);
    p.RandPerm();
    (B)(p,p,true);
    
    /* Normal fmadd semiring */
    typedef PlusTimesSRing<double,double> PTTF;
    
    
    double totalTime=0.0;
    int64_t perProcessMem = (512) / np; //Perlmutter CPU node
    
    int algCode = std::atoi(argv[3]);
   
    for (int i=0; i<ITERS; i++) {
        
        if (rank==0)
            PRINT("Iteration " + std::to_string(i)+"\n");

        auto stime = MPI_Wtime(); 
        
        switch(algCode) {
        
            case SPGEMMALG2D::MEM_EFFICIENT_2D_GPU:
            /*    MemEfficientSpGEMMg<PTTF, NT, DER>(
                                    A, B, //A, B
                                    1, //phases
                                    (NT)(1.0/10000.0), //hardThreshold
                                    (IT)1100, //selectNum
                                    (IT)1400, //recoverNum
                                    (NT)0.9, //recoverPct
                                    0, //kselectVersion
                                    perProcessMem, //perProcessMem
                                    LSPG_NSPARSE, //local_spgemm
                                    1); //nrounds */
               break; 
            
            case SPGEMMALG2D::MEM_EFFICIENT_2D:
                MemEfficientSpGEMM<PTTF, NT, DER>(
                                    A, B, //A, B
                                    1, //phases
                                    (NT)(1.0/10000.0), //hardThreshold
                                    (IT)1100, //selectNum
                                    (IT)1400, //recoverNum
                                    (NT)0.9, //recoverPct
                                    0, //kselectVersion
                                    2, //computational kernel
                                    perProcessMem //perProcessMem
                                    );
                break;
            case SPGEMMALG2D::DOUBLE_BUFF:
                Mult_AnXBn_DoubleBuff<PTTF, NT, DER>(A, B);
                break;
            case SPGEMMALG2D::SYNCH:
                Mult_AnXBn_Synch<PTTF, NT, DER>(A, B);
                break;
            case SPGEMMALG2D::OVERLAP:
                Mult_AnXBn_Overlap<PTTF, NT, DER>(A, B);
                break;
            case SPGEMMALG2D::BLOCKED:
            {
                /*
                int br=std::atoi(argv[5]);
                int bc=std::atoi(argv[6]);
                int bi=1;
                BlockSpGEMM bspgemm(A,B, br, bc, bi);
                IT rOffset; IT cOffset;
                while (bspgemm.hasNext()) {
                    bspgemm.getNextBlock<PTTF, NT, DER>(rOffset, cOffset);
                } */
                break;
            }
            default:
                std::cerr<<"Alg code "<<std::to_string(algCode)<<" not recongized"<<std::endl;
                exit(1);
    
        }
    
        
        auto etime = MPI_Wtime();
        if (i>0) //First iteration is slow
            totalTime += (etime-stime);
    
    } 
    
    double avgTime = totalTime / (ITERS-1);
    
    if (rank==0)
        PRINT("Avg time: " + std::to_string(avgTime) + "s\n");

}


enum SPGEMMALG3D {
    MEM_EFFICIENT_3D,
    SUMMA_3D
} typedef SPGEMMALG3D;

/*
 * Run a 3D SpGEMM algorithm
 * The value of the second command line argument determines which algorithm is run
 * 0: MemEfficientSpGEMM3D
 * 1: Mult_AnXBn_SUMMA3D
 */
void runSpGEMM3Dcpu(int argc, char ** argv) {
    
    /* MPI init */
    int np; int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    /* Comm Grid setup */
    std::shared_ptr<CommGrid> grid;
    grid.reset(new CommGrid(MPI_COMM_WORLD,0,0));
    
    
    /* Important types */
    typedef  int64_t IT;
    typedef  double NT;
    typedef SpDCCols <IT, NT> DER;
    typedef PlusTimesSRing<NT, NT> PTTF;
    
    /* Read in sparse matrix */
    SpParMat<IT, NT, DER> Atemp(grid); 
    std::string matName(argv[2]);
    Atemp.ParallelReadMM(matName, true, maximum<NT>());
    
    /* Setup 2D B */
    SpParMat<IT, NT, DER> Btemp(Atemp); 
    FullyDistVec<IT, NT> p(Atemp.getcommgrid());
    p.iota(Atemp.getnrow(), 0);
    p.RandPerm();
    (Btemp)(p,p,true);
    
    /* Convert A and B to 3D */
    int nLayers = std::atoi(argv[4]);
    SpParMat3D<IT,NT,DER> A(Atemp,nLayers, true);
    SpParMat3D<IT,NT,DER> B(Btemp,nLayers, false);

    double loadBalance = Atemp.LoadImbalance();
    if (rank==0)
        PRINT("Load balance: " + std::to_string(loadBalance));
    
    PRINT("Local nnz of A on rank " + std::to_string(rank) + ": " + std::to_string(A.seqptr()->getnnz()) );
    PRINT("Local nnz of B on rank " + std::to_string(rank) + ": " + std::to_string(B.seqptr()->getnnz()) );
    
    IT localFlops = EstimateFLOP<PTTF, IT, NT, NT, DER, DER>(*(A.GetLayerMat()), *(B.GetLayerMat()), false, false, true);

    
    PRINT("Local FLOPS on rank " + std::to_string(rank) + ": " + std::to_string(localFlops) );
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    double totalTime = 0.0;
    double perProcessMem = (512)/np;
    int algCode = std::atoi(argv[3]);
    
    for (int i=0;i<ITERS;i++) {
        
        if (rank==0)
            PRINT("Iteration " + std::to_string(i) + "\n");
        
        auto stime = MPI_Wtime();
        
        switch (algCode) {
        
            case SPGEMMALG3D::MEM_EFFICIENT_3D:
                MemEfficientSpGEMM3D<PTTF, NT, DER>(
                                    A, //A 
                                    B, //B
                                    1, //phases
                                    (NT)(1.0/10000.0), //hardThreshold
                                    (IT)1100, //selectNum
                                    (IT)1400, //recoverNum
                                    (NT)0.9, //recoverPct
                                    0, //kselectVersion
                                    2, //computational kernel
                                    perProcessMem //perProcessMem
                                    );
                break;
            case SPGEMMALG3D::SUMMA_3D:
                //This does not have an implementation on the combblas-gpu branch
                Mult_AnXBn_SUMMA3D<PTTF, NT, DER>(A,B);
                break;
            default:
                std::cerr<<"Algorithm code "<<algCode<<" not valid"<<std::endl;
                exit(1);
                break;

        }
    
        auto etime = MPI_Wtime();
        
        if (i>0) //First iteration is slow
            totalTime += (etime - stime);

    }
    
    double avgTime = totalTime / (ITERS-1);
    
    if (rank==0)
        PRINT("Avg time: " + std::to_string(avgTime) + "s\n");
    
    

}


int main(int argc, char ** argv) {
    
    /* ./binary <ALG-TYPE> <MAT-NAME> <ALG-CODE> <LAYERS> <br> <bc> 1 */

    assert(argc>3);
    
    std::string algType(argv[1]);
    
    if (algType.compare("2D")==0) {
        runSpGEMM2Dcpu(argc, argv);
    } else if (algType.compare("3D")==0) {
        runSpGEMM3Dcpu(argc, argv);
    } else if (algType.compare("1D")==0) {
        runSpGEMM1Dcpu(argc, argv);
    } else {
        std::cerr<<"Algorithm type "<<algType<<" not recognized"<<std::endl;
        exit(1);
    }
    
    
    return 0;

}

