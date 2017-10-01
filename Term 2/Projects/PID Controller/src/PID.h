#ifndef PID_H
#define PID_H

class PID {
public:
  /*
  * Errors
  */
  // Proportional error term
  double p_error;
  // Integral error term
  double i_error;
  // Derivative error term
  double d_error;

  /*
  * Coefficients
  */ 
  // Proportional coefficient
  double Kp;
  // Integral coefficient
  double Ki;
  // Derivative coefficient
  double Kd;

  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double Kp, double Ki, double Kd);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();
};

#endif /* PID_H */
