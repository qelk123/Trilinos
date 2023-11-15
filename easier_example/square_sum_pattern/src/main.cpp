#include <iostream>  
#include <chrono>  
#include <thread> 
#include <string>  
#include <fstream>  
#include <sstream>  
#include <string>  
#include <vector>  
#include <typeinfo>  

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
#include <Teuchos_LocalTestingHelpers.hpp>
#include <Tpetra_CrsMatrix_decl.hpp>
#include <Tpetra_CrsMatrixMultiplyOp.hpp>

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
    std::cout << "read over!\n";
  
    // Close the input file  
    input_file.close();

    return 0;
}


int main(int argc, char *argv[]) {
  std::string rhs_file_path = "../../data/b_medium_true.npy";

  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::tuple;
  using scalar_type = double;
  using local_ordinal_type = Tpetra::MultiVector<double, int, long long>::local_ordinal_type;
  using global_ordinal_type = Tpetra::MultiVector<double, int, long long>::global_ordinal_type;
  using node_type = Tpetra::MultiVector<>::node_type;
  typedef size_t global_size_t; 
  typedef ::Tpetra::Map<local_ordinal_type, global_ordinal_type, node_type> map_type;
  typedef Teuchos::ScalarTraits<scalar_type> ST;

  using crs_matrix_type = Tpetra::CrsMatrix<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;
  using multivec_type = Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;
  using operator_type = Tpetra::Operator<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;
  using vec_type = Tpetra::Vector<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;
  typedef typename ST::magnitudeType Mag;
  Tpetra::ScopeGuard tpetraScope(&argc, &argv);
  {
    RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
    const size_t myRank = comm->getRank();
    const size_t numProcs = comm->getSize();
    RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
    out->setOutputToRootOnly(0);

    cnpy::NpyArray arr = cnpy::npy_load(rhs_file_path);  
    double* data = arr.data<double>();
    RCP<const map_type> map(new map_type(arr.shape[0], 0, comm));
    RCP<vec_type> X = rcp(new vec_type(map));  

    global_ordinal_type start_global_index = map->getMinGlobalIndex();
    for (size_t i = 0; i < X->getLocalLength(); ++i) {  
      X->replaceLocalValue(i, data[start_global_index + i]);
    }

    vec_type X_copy (*X, Teuchos::Copy);
    Teuchos::Array<scalar_type> dots(1);

    nvtxRangePushA("range");
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();  
    for (int i = 0; i < 1; i ++) {
      X_copy.dot(X_copy, dots());
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();  
    nvtxRangePop(); 
    std::chrono::duration<double> elapsed_seconds = end - start;  
    std::cout << "Elapsed time from cpu side: " << elapsed_seconds.count() << "s\n"; 

    std::cout << "rank" << myRank << "reduce result :" << dots[0] << std::endl;
  }
}
