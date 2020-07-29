#ifndef LKlevenbergMarquardt_h
#define LKlevenbergMarquardt_h

#include <cstring>
#include <Eigen/Dense>


///////////////////////////////////////////////////////////////////////
// \brief newton algorithm : solves non linear system.
// We use in this implementation the recommandation of the "Multiple View Geometry", Hartley & Zisserman, 2nd edition, page 597


class LevenbergMarquardt{

public :

  ///////////////////////////////////////////////////////////////////////
  /// constructor
  LevenbergMarquardt(){}

  ///////////////////////////////////////////////////////////////////////
  /// destructor
  virtual ~LevenbergMarquardt(){}

  ///////////////////////////////////////////////////////////////////////
  /// This function solves a non-linear system. 
  /// the x vector is the unknown
  /// The vector X corresponds to the state we want to reach : we want X=f(a,b).
  /// This function returns 1 if an acceptable solution has been found before the nbMaxIteration iterations.
  /// This algorithm is detailed in the "Multiple View Geometry", Hartley & Zisserman, 2nd edition, page 605.
  int minimize(Eigen::VectorXd &a, const Eigen::VectorXd &X, const unsigned int nbMaxIteration, const bool statVerbose=true);

  ///////////////////////////////////////////////////////////////////////
  /// print the minimization statistics
  std::string statistics(const std::string &convergence, const Eigen::VectorXd &initErr, const Eigen::VectorXd &finalErr)const;

protected :

  ///////////////////////////////////////////////////////////////////////
  /// error function to minimize : should be "surcharged" in the daugther class
  virtual Eigen::VectorXd error(const Eigen::VectorXd &x) const = 0;

  ///////////////////////////////////////////////////////////////////////
  /// This function computes the jacobian of 'a'.
  /// This function is not optimized (a lot of 0 are computed for nothing) but can be adapted to every partitioned data.
  /// This algorithm is detailed in the "Multiple View Geometry", Hartley & Zisserman, 2nd edition, page 602-605.
  void levenbergMarquardtJacobian_A(const Eigen::VectorXd &x, Eigen::MatrixXd &jacobian);

  ///////////////////////////////////////////////////////////////////////
  /// return true if the error has decreased, false if the error has increased
  bool levenbergMarquardtCheckImprovement(const Eigen::VectorXd &epsilonBefore, const Eigen::VectorXd &epsilonAfter) const;

  ///////////////////////////////////////////////////////////////////////
  /// print some statistics of the error vector
  void errorStat(const Eigen::VectorXd &vecError, const Eigen::VectorXd &targetError) const; 



protected :

  ///////////////////////////////////////////////////////////////////////
  /// statistic data: number of iterations 
  unsigned int statisticNbIterations;

  ///////////////////////////////////////////////////////////////////////
  /// statistic data: number of Jacobian computation
  unsigned int statisticNbJacobianComputations;

  ///////////////////////////////////////////////////////////////////////
  /// boolean: print the satistic or not
  bool verbose;

 
};

#endif
