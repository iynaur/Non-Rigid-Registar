#ifndef OPTIMAL_NONRIGID_ICP_H
#define OPTIMAL_NONRIGID_ICP_H

#include <vtkCell.h>
#include <vtkCellLocator.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

#include <vector>

#include <memory>

typedef vtkSmartPointer<vtkPolyData> Mesh;

typedef Mesh Template;
typedef Mesh Target;
typedef std::pair<int, int> Edge;
typedef std::shared_ptr<std::vector<Edge> > Edges;
typedef vtkSmartPointer<vtkPoints> Vertices;
typedef std::shared_ptr<std::vector<float> > Weights;

class OptimalNonrigidICP {
 public:
  OptimalNonrigidICP(Template _template, Target _target)
      : _template(_template), _target(_target) {}
  ~OptimalNonrigidICP() {}

  void init() {
    edgesInit();
    verticesInit();
    nearestSearchInit();
  }

  void initCompute() {
    correspondencesInit();
    weightsInit();
  }

  void edgesInit() {
    if (_edges == NULL) _edges.reset(new std::vector<Edge>);

    for (int i = 0; i < _template->GetNumberOfCells(); ++i) {
      vtkCell* cell = _template->GetCell(i);
      int a = cell->GetPointId(0);
      int b = cell->GetPointId(1);
      int c = cell->GetPointId(2);

      _edges->push_back(Edge(a, b));
      _edges->push_back(Edge(b, c));
      _edges->push_back(Edge(c, a));
    }
  }

  void nearestSearchInit() {
    _cellLocator = vtkSmartPointer<vtkCellLocator>::New();
    _cellLocator->SetDataSet(_target);
    _cellLocator->BuildLocator();
  }

  void verticesInit() { _vertices = _template->GetPoints(); }

  void correspondencesInit() {
    if (_correspondences == NULL) _correspondences = Vertices::New();

    _correspondences->SetNumberOfPoints(_vertices->GetNumberOfPoints());
    for (int i = 0; i < _vertices->GetNumberOfPoints(); i++) {
      double testPoint[3];
      _vertices->GetPoint(i, testPoint);
      double closestPoint[3];
      double closestPointDist2;
      vtkIdType cellId;
      int subId;
      _cellLocator->FindClosestPoint(testPoint, closestPoint, cellId, subId,
                                     closestPointDist2);
      _correspondences->SetPoint(i, closestPoint);
    }
  }

  void weightsInit() {
    if (_weights == NULL) _weights.reset(new std::vector<float>());

    _weights->resize(_vertices->GetNumberOfPoints());
    for (int i = 0; i < _weights->size(); ++i) (*_weights)[i] = 1.0f;
  }

  int compute(float alpha, float gamma) {
    // To do nonrigid icp registration

    int n = _vertices->GetNumberOfPoints();
    int m = _edges->size();

    Eigen::SparseMatrix<float> A(4 * m + n, 4 * n);
    std::cout << "The matrix A is of size " << A.rows() << "x" << A.cols()
              << std::endl;

    std::vector<Eigen::Triplet<float> > alpha_M_G;
    for (int i = 0; i < m; ++i) {
      Edge edge = (*_edges)[i];
      int a = edge.first;
      int b = edge.second;

      for (int j = 0; j < 3; j++)
        alpha_M_G.push_back(Eigen::Triplet<float>(i * 4 + j, a * 4 + j, alpha));
      alpha_M_G.push_back(
          Eigen::Triplet<float>(i * 4 + 3, a * 4 + 3, alpha * gamma));

      for (int j = 0; j < 3; j++)
        alpha_M_G.push_back(
            Eigen::Triplet<float>(i * 4 + j, b * 4 + j, -alpha));
      alpha_M_G.push_back(
          Eigen::Triplet<float>(i * 4 + 3, b * 4 + 3, -alpha * gamma));
    }
    std::cout << "alpha_M_G calculated!" << std::endl;

    std::vector<Eigen::Triplet<float> > W_D;
    for (int i = 0; i < n; ++i) {
      double xyz[3];
      _vertices->GetPoint(i, xyz);

      float weight = (*_weights)[i];

      for (int j = 0; j < 3; ++j)
        W_D.push_back(
            Eigen::Triplet<float>(4 * m + i, i * 4 + j, weight * xyz[j]));
      W_D.push_back(Eigen::Triplet<float>(4 * m + i, i * 4 + 3, weight));
    }
    std::cout << "W_D calculated!" << std::endl;

    std::vector<Eigen::Triplet<float> > _A = alpha_M_G;
    _A.insert(_A.end(), W_D.begin(), W_D.end());
    std::cout << "_A calculated!" << std::endl;

    A.setFromTriplets(_A.begin(), _A.end());
    std::cout << "A calculated!" << std::endl;

    Eigen::MatrixX3f B = Eigen::MatrixX3f::Zero(4 * m + n, 3);
    std::cout << "The matrix B is of size " << B.rows() << "x" << B.cols()
              << std::endl;
    for (int i = 0; i < n; ++i) {
      double xyz[3];
      _correspondences->GetPoint(i, xyz);

      float weight = (*_weights)[i];
      for (int j = 0; j < 3; j++) {
        B(4 * m + i, j) = weight * xyz[j];
      }
    }
    std::cout << "B calculated!" << std::endl;

    Eigen::SparseMatrix<float> ATA =
        Eigen::SparseMatrix<float>(A.transpose()) * A;
    std::cout << "ATA calculated!" << std::endl;
    Eigen::MatrixX3f ATB = Eigen::SparseMatrix<float>(A.transpose()) * B;
    std::cout << "ATB calculated!" << std::endl;

    Eigen::SparseLU<Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<int> >
        solver;
    // Compute the ordering permutation vector from the structural pattern of A
    solver.analyzePattern(ATA);
    std::cout << "ATA analyzed!" << std::endl;
    // Compute the numerical factorization
    solver.factorize(ATA);
    std::cout << "ATA factorized!" << std::endl;
    if (solver.info() != Eigen::Success) {
      std::cerr << "Decomposition failed" << std::endl;
      return 1;
    }
    Eigen::MatrixX3f X = solver.solve(ATB);
    std::cout << "X calculated!" << std::endl;

    Eigen::Matrix3Xf XT = X.transpose();
    for (int i = 0; i < n; ++i) {
      double xyz[3];
      _vertices->GetPoint(i, xyz);
      Eigen::Vector4f point(xyz[0], xyz[1], xyz[2], 1.0f);
      Eigen::Vector3f point_transformed = XT.block<3, 4>(0, 4 * i) * point;
      _vertices->SetPoint(i, point_transformed[0], point_transformed[1],
                          point_transformed[2]);

      double txyz[3];
      _correspondences->GetPoint(i, txyz);

      // if ( i < 10) std::cout << XT.block<3, 4>(0, 4*i) << std::endl;
      if (i < 10)
        std::cout << xyz[0] << "," << xyz[1] << "," << xyz[2] << " -> "
                  << point_transformed[0] << " " << point_transformed[1] << " "
                  << point_transformed[2] << " -> " << txyz[0] << "," << txyz[1]
                  << "," << txyz[2] << std::endl;
    }

    return 0;
  }

 protected:
  Template _template;
  Target _target;

  Edges _edges;
  Vertices _vertices;

  Vertices _correspondences;
  Weights _weights;

 private:
  vtkSmartPointer<vtkCellLocator> _cellLocator;
};

#endif
