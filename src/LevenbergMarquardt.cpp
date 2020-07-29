#include "LevenbergMarquardt.hpp"

#include <cmath>
#include <iostream> 
#include <fstream> 



///////////////////////////////////////////////////////////////////////
/// This function solves a non-linear system. 
/// the a vector is the unknown
/// the b vector is the const data
/// The vector X corresponds to the state we want to reach : we want X=f(a,b).
/// This function returns 1 if an acceptable solution has been found before the nbMaxIteration iterations.
/// This algorithm is detailed in the "Multiple View Geometry", Hartley & Zisserman, 2nd edition, page 605.
int LevenbergMarquardt::minimize(Eigen::VectorXd &a, const Eigen::VectorXd &X, const unsigned int nbMaxIteration, const bool statVerbose)
{
  // verbose
  verbose = statVerbose;

  // statistic reset
  statisticNbIterations = statisticNbJacobianComputations = 0;
  Eigen::VectorXd initialError(error(a)-X);

  // Jacobian matrix for 'a'
  Eigen::MatrixXd J(X.size(),a.size());

  // compute the jacobian
  levenbergMarquardtJacobian_A(a,J);

  // lambda parameter : decides if LM is a Gauss-Newton method or a gradient descent method.
  // lambda = 1.0e-3 x average of diag(N=J^T.J)  is a typical starting value.
  Eigen::MatrixXd N(J.transpose()*J);
  double average = 0.0;
  for(unsigned int i=0; i<N.rows(); ++i)
    average += N(i,i);
  double lambda = 1.0e-3 * average/(double)N.rows();

  // start the iterative process
  for(unsigned int iter=0; iter<nbMaxIteration; ++iter)
    {
      // compute the jacobian
      levenbergMarquardtJacobian_A(a,J);

      // error vector
      Eigen::VectorXd epsilon(X-error(a));

      // iteration to update lambda if not adapted
      unsigned int counter = 0;
      bool accepted = false; // true if an iteration makes improvement

      do{
        // find delta
	Eigen::VectorXd delta_a(a.size());

	Eigen::MatrixXd Id(J.transpose()*J);
	Id.setIdentity();
	delta_a = (J.transpose()*J + lambda*Id).jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(J.transpose()*epsilon);

        // update parameters
        Eigen::VectorXd a2(a + delta_a);

        // compute new error vector
        Eigen::VectorXd epsilon2(X-error(a2));

        // compare error
        if(levenbergMarquardtCheckImprovement(epsilon, epsilon2) == true) // if better
        {
          lambda /= 10.0;
          a = a2;
          accepted = true;
        }else
        {
          lambda *= 10.0;
          accepted = false;
        }

        // counter
        ++counter;
        if(counter > 100){
          if(verbose) std::cout << statistics("true",initialError,(error(a)-X));
          return 1;
        }

      }while(accepted == false); // end of an iteration

      // statistic reset
      ++statisticNbIterations;

     } // end of all iteration

  // statistic
  if(verbose) std::cout << statistics("false",initialError,(error(a)-X));

  return 0;
}


///////////////////////////////////////////////////////////////////////
/// This function computes the jacobian of 'a'.
/// This function is not optimized (a lot of 0 are computed for nothing) but can be adapted to every partitioned data.
/// This algorithm is detailed in the "Multiple View Geometry", Hartley & Zisserman, 2nd edition, page 602-605.
void LevenbergMarquardt::levenbergMarquardtJacobian_A(const Eigen::VectorXd &a, Eigen::MatrixXd &jacobian_A)
{
  // statistic
  ++statisticNbJacobianComputations;

  // Jacobian computation
  Eigen::VectorXd aTmp(a);
  Eigen::VectorXd Xcurrent = error(a);

  // can be computed column per column
  for(unsigned int j=0; j<a.size(); ++j)
  {
    double delta = std::max(std::min(std::fabs(a[j])*1.0e-4,1.0e-6),1.0e-13); // cf multiple view geometry, 2nd ed., page 602

    aTmp = a;
    aTmp[j] += delta;                  // P'[j] = P[j] + delta

    Eigen::VectorXd Xtmp = error(aTmp);  // f(P')
    Xtmp = (Xtmp - Xcurrent) / delta;  // for every line i : dX_i/da_j

    jacobian_A.col(j) = Xtmp;
  }
}


///////////////////////////////////////////////////////////////////////
/// return true if the error has decreased, false if the error has increased
bool LevenbergMarquardt::levenbergMarquardtCheckImprovement(const Eigen::VectorXd &epsilonBefore, const Eigen::VectorXd &epsilonAfter) const
{
  double sumBefore = 0.0;
  double sumAfter  = 0.0;

  for(unsigned int i=0; i<epsilonBefore.size(); ++i)
   {
     sumBefore += epsilonBefore[i] * epsilonBefore[i]; //fabs(epsilonBefore[i]);
     sumAfter  += epsilonAfter[i]  * epsilonAfter[i];  //fabs(epsilonAfter[i]);
   }

  return sumBefore > sumAfter;
}


///////////////////////////////////////////////////////////////////////
/// return true if the error has decreased, false if the error has increased
void LevenbergMarquardt::errorStat(const Eigen::VectorXd &vecError, const Eigen::VectorXd &targetError)const
{
  double max = 0.0;
  double average = 0.0;
  double mean = 0.0;

  for(unsigned int i=0; i< vecError.size(); ++i)
    {
      if(fabs(vecError(i)-targetError(i)) > max) max = fabs(vecError[i]);

      average += fabs(vecError(i)-targetError(i));
    }

  average = average / vecError.size();

  for(unsigned int i=0; i< vecError.size(); ++i)
    mean +=  fabs(fabs(vecError(i)-targetError(i))-average);

  mean = mean / vecError.size();

  std::cout << "    max     : " << max << std::endl; 
  std::cout << "    average : " << average << std::endl; 
  std::cout << "    mean    : " << mean << std::endl; 
  std::cout << std::endl; 
}


///////////////////////////////////////////////////////////////////////
/// print the minimization statistics
std::string LevenbergMarquardt::statistics(const std::string &convergence, const Eigen::VectorXd &initErr, const Eigen::VectorXd &finalErr)const
{
  std::ostringstream stat;
  stat << "    convergence reached  : " << convergence << std::endl
       << "    iterations           : " << statisticNbIterations << std::endl
       << "    jacobian computation : " << statisticNbJacobianComputations << std::endl
       << "    initial error        : " << initErr.transpose() << std::endl
       << "    final error          : " << finalErr.transpose() << std::endl;
 
  return stat.str();
}



