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


int main() {


  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::tuple;
  // typedef Tpetra::Map<long long, long long>::global_ordinal_type LO;
  // typedef Tpetra::Map<long long, long long>::global_ordinal_type GO;
  using Scalar = double;
  using LO = Tpetra::MultiVector<double, int, long long>::local_ordinal_type;
  // using local_ordinal_type = Tpetra::MultiVector<>::local_ordinal_type;
  using GO = Tpetra::MultiVector<double, int, long long>::global_ordinal_type;
  using global_ordinal_type = Tpetra::MultiVector<>::global_ordinal_type;
  using Node = Tpetra::MultiVector<>::node_type;
  typedef size_t global_size_t;
  // using node_type = Tpetra::MultiVector<>::node_type;
  // std::cout << "local_ordinal_type: " << typeid(local_ordinal_type).name() << std::endl;  
  // std::cout << "global_ordinal_type: " << typeid(global_ordinal_type).name() << std::endl;  
  typedef ::Tpetra::Map<LO, GO, Node> map_type;

  // using crs_matrix_type = Tpetra::CrsMatrix<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;
  // using multivec_type = Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;
  // using operator_type = Tpetra::Operator<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;
  // using row_matrix_type = Tpetra::RowMatrix<>;
  // using vec_type = Tpetra::Vector<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;

  // using problem_type = Belos::LinearProblem<scalar_type, multivec_type, operator_type>;
  // using solver_type = Belos::SolverManager<scalar_type, multivec_type, operator_type>;

  typedef Tpetra::CrsMatrix<Scalar,LO,GO,Node> MAT;
  typedef Tpetra::Operator<Scalar,LO,GO,Node> OP;
  typedef Teuchos::ScalarTraits<Scalar> ST;
  typedef Tpetra::MultiVector<Scalar,LO,GO,Node> MV;
  typedef typename ST::magnitudeType Mag;
  const size_t THREE = 3;
  const global_size_t INVALID = (global_size_t)-1;
  RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
  const size_t myImageID = comm->getRank();
  // RCP<const Map<LO,GO,Node> > map = createContigMapWithNode<LO,GO,Node>(INVALID,THREE,comm);

  /* Create the identity matrix, three rows per proc */
  typedef Tpetra::MatrixMarket::Reader<MAT> reader_type;
  std::string input_file_string;
  read_mtx_file("/home/v-yinuoliu/code/tvms/matrix2388.mtx", input_file_string);
  // read_mtx_file("/home/v-yinuoliu/code/tvms/output_matrix_medium_true.mtx", input_file_string);
  std::istringstream matrixFile(input_file_string);
  RCP<MAT> matrix = reader_type::readSparse(matrixFile, comm);
  // cnpy::NpyArray arr = cnpy::npy_load("/home/v-yinuoliu/code/tvms/b_medium_true.npy");  
  // double* data = arr.data<double>();
  // RCP<const map_type> map(new map_type(arr.shape[0], 0, comm));
  global_ordinal_type row_num = 1102824;
  RCP<const map_type> map(new map_type(row_num, 0, comm));
  

  // RCP<OP> AOp;
  // RCP<MAT> A = rcp(new MAT(map,1));
  // A->insertGlobalValues(3*myImageID,  tuple<GO>(3*myImageID  ), tuple<Scalar>(ST::one()) );
  // A->insertGlobalValues(3*myImageID+1,tuple<GO>(3*myImageID+1), tuple<Scalar>(ST::one()) );
  // A->insertGlobalValues(3*myImageID+2,tuple<GO>(3*myImageID+2), tuple<Scalar>(ST::one()) );
  // A->fillComplete();
  // AOp = A;

  MV X(map,1), Y(map,1), Z(map,1);
  const Scalar alpha = (double) 1.0,
                beta = (double) 0.0;

  // for (global_ordinal_type i = 0; i < arr.shape[0]; ++i) {  
  //   X.replaceGlobalValue(i, 0, data[i]);
  for (global_ordinal_type i = 0; i < row_num; ++i) {  
    X.replaceGlobalValue(i, 0, 1.0);
    Y.replaceGlobalValue(i, 0, 0.0);
  }

  // X.randomize();
  // Y.randomize();
  // // Keep copies for later testing of CrsMatrixMultiplyOp
  MV X_copy (X, Teuchos::Copy);
  MV Y_copy (Y, Teuchos::Copy);

  // // Z = alpha*X + beta*Y
  // Z.update(alpha,X,beta,Y,ST::zero());
  // // test the action: Y = alpha*I*X + beta*Y = alpha*X + beta*Y = Z
  // AOp->apply(X,Y,Teuchos::NO_TRANS,alpha,beta);

  // mfh 07 Dec 2018: Little test for CrsMatrixMultiplyOp; it
  // doesn't get tested much elsewhere.  (It used to be part of
  // CrsMatrix's implementation, so it got more exercise before.)
  Tpetra::CrsMatrixMultiplyOp<Scalar, Scalar, LO, GO, Node> multOp (matrix);
  for (int i = 0; i < 1000; i ++) {
    multOp.apply (X_copy, Y_copy, Teuchos::NO_TRANS, alpha, beta);
  }

  Teuchos::Array<Mag> normY(1), normZ(1);
  Z.norm1(normZ());
  Y.norm1(normY());

  Teuchos::ArrayRCP<const double> x_values = Y_copy.get1dView();  
    
  // int num_elements = map->NumMyElements();  
  for (int i = 0; i < 10; ++i) {  
      std::cout << "x_values Element " << i << ": " << x_values[i] << std::endl;  
  }  
  // if (ST::isOrdinal) {
  //   TEST_COMPARE_ARRAYS(normY,normZ);
  // } else {
  //   TEST_COMPARE_FLOATING_ARRAYS(normY,normZ,2.0*testingTol<Mag>());
  // }

  // Teuchos::Array<Mag> normYcopy(1);
  // Y_copy.norm1 (normYcopy ());
  // if (ST::isOrdinal) {
  //   TEST_COMPARE_ARRAYS(normYcopy,normZ);
  // } else {
  //   TEST_COMPARE_FLOATING_ARRAYS(normYcopy,normZ,2.0*testingTol<Mag>());
  // }
}
