/* This example demonstrate the solution of a linear system with a preconditioner Krylov solver
 * with the help of the packages Belos and Ifpack2.
 */

#include <iostream>  
#include <chrono>  
#include <thread> 
#include <string>  
#include <fstream>  
#include <sstream>  
#include <string>  
#include <vector>  
#include <typeinfo>  

#include "utils.hpp"
#include "cnpy.h"

#include <cstdlib>
#include <cuda_runtime.h>  
#include <nvToolsExt.h>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosTypes.hpp>

#include <Ifpack2_Factory.hpp>
#include <Ifpack2_Preconditioner.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_Vector.hpp>

#include <BelosSolverFactory.hpp>
#include <MatrixMarket_Tpetra.hpp>
#include <Kokkos_Core.hpp>  


int read_mtx_file(std::string file_name, std::string & output_str) {

    std::ifstream input_file(file_name);  
    if (!input_file.is_open()) {  
        std::cerr << "Error opening file" << std::endl;  
        return 1;  
    }  
    // Read the input file into a std::string  
    std::stringstream buffer;  
    buffer << input_file.rdbuf();  
    output_str = buffer.str();  
    // Close the input file  
    input_file.close();

    return 0;
}


// export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
int main(int argc, char *argv[]) {
  std::string matrix_file_path = "../../data/output_matrix_medium_true.mtx";
  std::string rhs_file_path = "../../data/b_medium_true.npy";


  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::tuple;
  using scalar_type = double;
  using local_ordinal_type = Tpetra::MultiVector<double, int, long long>::local_ordinal_type;
  using global_ordinal_type = Tpetra::MultiVector<double, int, long long>::global_ordinal_type;
  using node_type = Tpetra::MultiVector<>::node_type;
  std::cout << "local_ordinal_type: " << typeid(local_ordinal_type).name() << std::endl;  
  std::cout << "global_ordinal_type: " << typeid(global_ordinal_type).name() << std::endl;  
  typedef ::Tpetra::Map<local_ordinal_type, global_ordinal_type, node_type> map_type;

  using crs_matrix_type = Tpetra::CrsMatrix<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;
  using multivec_type = Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;
  using operator_type = Tpetra::Operator<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;
  using row_matrix_type = Tpetra::RowMatrix<>;
  using vec_type = Tpetra::Vector<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;

  using problem_type = Belos::LinearProblem<scalar_type, multivec_type, operator_type>;
  using solver_type = Belos::SolverManager<scalar_type, multivec_type, operator_type>;

  scalar_type tol = 1.0e-5;
  Tpetra::ScopeGuard tpetraScope(&argc, &argv);
  {
    // Create MPI communicator via Tpetra and obtain local MPI rank and the
    // total size of the MPI communicator
    RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
    const size_t myRank = comm->getRank();
    const size_t numProcs = comm->getSize();

    // Create an output stream
    RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
    out->setOutputToRootOnly(0);

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    *out << ">> I. Create linear system A*x=b." << std::endl;
    // load A and b from file
    typedef Tpetra::MatrixMarket::Reader<crs_matrix_type> reader_type;
    std::string input_file_string;
    read_mtx_file(matrix_file_path, input_file_string);
    std::istringstream matrixFile(input_file_string);
    RCP<crs_matrix_type> matrix = reader_type::readSparse(matrixFile, comm);
    cnpy::NpyArray arr = cnpy::npy_load(rhs_file_path);  
    double* data = arr.data<double>();


    // // Create a Tpetra Vector for the right-hand side (RHS) and fill it with your custom values  
    RCP<const map_type> map(new map_type(arr.shape[0], 0, comm));
    global_ordinal_type start_global_index = map->getMinGlobalIndex();
    RCP<vec_type> rhs = rcp(new vec_type(map));  
    RCP<vec_type> x = rcp(new vec_type(map));

    std::cout << "start global index for rank " << myRank << " is:" << start_global_index << "\n"; 
    std::cout << "local vector length for rank " << myRank << " is:" << x->getLocalLength() << "\n"; 

    // Fill your custom RHS vector here
    for (size_t i = 0; i < x->getLocalLength(); ++i) {  
      rhs->replaceLocalValue(i, data[start_global_index + i]);
      x->replaceLocalValue(i, 0.0);
    }
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    *out << ">> II. Create a " << "GMRES solver from the Belos package." << std::endl;

    // Create Belos iterative linear solver
    RCP<solver_type> solver = Teuchos::null;
    RCP<ParameterList> solverParams = rcp(new ParameterList());
    {
      // int verbLevel = Belos::Errors + Belos::Warnings + Belos::FinalSummary;
      solverParams->set("Num Blocks", 1);               //restarter num
      solverParams->set("Block Size", 1);               // Blocksize to be used by iterative solver
      solverParams->set("Maximum Iterations", 1000);    // Maximum number of iterations allowed
      solverParams->set("Maximum Restarts", 1000);      // Maximum number of restarts allowed
      solverParams->set("Convergence Tolerance", tol);  // Relative convergence tolerance requested
      solverParams->set("Orthogonalization", "IMGS");
    }

    // Set up the linear problem to solve.
    RCP<problem_type> problem = Teuchos::null;
    {
      problem = rcp(new problem_type(matrix, x, rhs));
      problem->setProblem();
      solver = \
        Teuchos::rcp(new Belos::PseudoBlockGmresSolMgr<scalar_type, multivec_type, operator_type>(problem, solverParams));
    }

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    *out << ">> III. Solve the linear system." << std::endl;

    // Solve the linear system.
    {
      std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();  
      nvtxRangePushA("solver");
      Belos::ReturnType solveResult = solver->solve();
      nvtxRangePop(); 
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();  
      std::chrono::duration<double> elapsed_seconds = end - start;  
      std::cout << "Elapsed time from cpu side: " << elapsed_seconds.count() << "s\n"; 
      if (solveResult == Belos::Unconverged)
      {
        *out << "iterations to an achieved tolerance of " << solver->achievedTol() << ". Belos did not converge in " << solver->getNumIters() << " iterations." << std::endl;
      }
      else
      {
        *out << "Belos converged in " << solver->getNumIters() << ", solver->isLOADetected(): " << 
            solver->isLOADetected()
            << ", iterations to an achieved tolerance of " << solver->achievedTol()
            << " (< tol = " << tol << ")." << std::endl;
      }
      Teuchos::ArrayRCP<const double> x_values = x->get1dView();  
      
      //dump some result
      for (int i = 0; i < 10; ++i) {  
          std::cout << "rank" << myRank << "x_values Element " << i << ": " << x_values[i] << std::endl;  
      }  
      std::vector<size_t> shape = {x->getLocalLength()};  
        
      std::string file_name = "trilinos_res_" + std::to_string(numProcs) + "_" + std::to_string(myRank) + ".npy";
      cnpy::npy_save(file_name, x_values.getRawPtr(), shape, "w");
    }
    return EXIT_SUCCESS;
  }
}
