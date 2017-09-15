#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:
    ///* initially set to false, set to true in first call of ProcessMeasurement
    bool is_initialized_;

    ///* if this is false, laser measurements will be ignored (except for init)
    bool use_laser_;

    ///* if this is false, radar measurements will be ignored (except for init)
    bool use_radar_;

    ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    VectorXd x_;

    ///* state covariance matrix
    MatrixXd P_;

    ///* predicted sigma points matrix
    MatrixXd Xsig_pred_;

    ///* time when the state is true, in us
    long long time_us_;

    ///* previous timestamp
    long long previous_timestamp_;

    ///* Process noise standard deviation longitudinal acceleration in m/s^2
    double std_a_;

    ///* Process noise standard deviation yaw acceleration in rad/s^2
    double std_yawdd_;

    ///* Laser measurement noise standard deviation position1 in m
    double std_laspx_;

    ///* Laser measurement noise standard deviation position2 in m
    double std_laspy_;

    ///* Radar measurement noise standard deviation radius in m
    double std_radr_;

    ///* Radar measurement noise standard deviation angle in rad
    double std_radphi_;

    ///* Radar measurement noise standard deviation radius change in m/s
    double std_radrd_ ;

    ///* Weights of sigma points
    VectorXd weights_;

    ///* State dimension
    int n_x_;

    ///* Augmented state dimension
    int n_aug_;

    ///* Sigma point spreading parameter
    double lambda_;


    ///* Laser H matrix
    MatrixXd H_laser_;
  
    ///* Laser measurement covariance matrix
    MatrixXd R_laser_;


    /**
    * Constructor
    */
    UKF();

    /**
    * Destructor
    */
    virtual ~UKF();

    /**
    * ProcessMeasurement
    * @param measurement_pack The latest measurement data of either radar or laser
    */
    void ProcessMeasurement(MeasurementPackage measurement_pack);

    /**
    * Prediction Predicts sigma points, the state, and the state covariance
    * matrix
    * @param delta_t Time between k and k+1 in s
    */
    void Prediction(double delta_t);

    /**
    * Augment sigma points.
    */
    MatrixXd AugmentedSigmaPoints();

    /**
    * Predict sigma points.
    */
    void SigmaPointPrediction(double delta_t);

    /**
    * Predict the state and covariance matrix.
    */
    void PredictMeanAndCovariance();

    /**
    * Updates the state and the state covariance matrix using a laser measurement
    * @param measurement_pack The measurement at k+1
    */
    void UpdateLidar(MeasurementPackage measurement_pack);

    /**
    * Updates the state and the state covariance matrix using a radar measurement
    * @param measurement_pack The measurement at k+1
    */
    void UpdateRadar(MeasurementPackage measurement_pack);

    /**
    * Use sigma points to predict the radar measurement.
    */
    void PredictRadarMeasurement(int n_z, MatrixXd* Z_out, VectorXd* z_out, MatrixXd* S_out);

    /**
    * Use measurement prediction to update the state and covariance matrix.
    */
    void UpdateState(int n_z, MeasurementPackage measurement_pack, MatrixXd Zsig, VectorXd z_pred, MatrixXd S);
};

#endif /* UKF_H */
