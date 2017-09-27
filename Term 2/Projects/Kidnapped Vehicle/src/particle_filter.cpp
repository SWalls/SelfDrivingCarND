/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *  Modified on: Sep 27, 2017
 *      Author: Soeren Walls
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles.
	num_particles = 100;

	// Randomizer
	default_random_engine gen;
	
	// Create normal (Gaussian) distributions for x, y, and theta.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Initialize all particles to first position (based on estimates of 
	// x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	for (int i = 0; i < num_particles; ++i) {
		double sample_x, sample_y, sample_theta;
		
		// Sample from these normal distributions.
        sample_x = dist_x(gen);
        sample_y = dist_y(gen);
        sample_theta = dist_theta(gen);
		
		// Create a new particle.
		Particle particle;
		particle.id = i;
		particle.x = sample_x;
		particle.y = sample_y;
		particle.theta = sample_theta;
		particle.weight = 1.0;

		particles.push_back(particle);
		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// New values for particles.
	float new_x;
	float new_y;
	float new_theta;

	// Randomizer
	default_random_engine gen;

	// Add measurements to each particle and add random Gaussian noise.
	for (int i = 0; i < num_particles; ++i) {
		Particle particle = particles[i];

		// Calculate new particle values.
		if (yaw_rate == 0) {
			new_x = particle.x + velocity * delta_t * cos(particle.theta);
			new_y = particle.y + velocity * delta_t * sin(particle.theta);
			new_theta = particle.theta;
		} else {
			new_x = particle.x + (velocity/yaw_rate) * (sin(particle.theta + (yaw_rate*delta_t)) - sin(particle.theta));
			new_y = particle.y + (velocity/yaw_rate) * (cos(particle.theta) - cos(particle.theta + (yaw_rate*delta_t)));
			new_theta = particle.theta + yaw_rate * delta_t;
		}
		
		// Set up gaussian distributions for noise.
		normal_distribution<double> dist_x(new_x, std_pos[0]);
		normal_distribution<double> dist_y(new_y, std_pos[1]);
		normal_distribution<double> dist_theta(new_theta, std_pos[2]);

		// Assign new particle values.
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a multvariate Gaussian distribution.
	for (int i = 0; i < num_particles; ++i) {
		// Reset particle weight
		particles[i].weight = 1.0;
		std::vector<int> associations;
		std::vector<double> sense_x;
		std::vector<double> sense_y;

		// Calculate new weight using data association.
		for (int j = 0; j < observations.size(); ++j) {
			LandmarkObs obs = observations[j];
			LandmarkObs obsT;

			// Transform from vehicle coordinates to map coordinates.
			obsT.x = particles[i].x + (obs.x * cos(particles[i].theta)) + (obs.y * sin(particles[i].theta));
			obsT.y = particles[i].y + (obs.x * sin(particles[i].theta)) - (obs.y * cos(particles[i].theta));

			// Associate transformed observation with landmark identifiers.
			float min_distance = 9999999999;
			int nearest_landmark = -1;
			for (int k = 0; k < map_landmarks.landmark_list.size(); ++k) {
				float map_x = map_landmarks.landmark_list[k].x_f;
				float map_y = map_landmarks.landmark_list[k].y_f;
				float distance = dist(obsT.x, obsT.y, map_x, map_y);
				if (distance < min_distance && distance <= sensor_range) {
					min_distance = distance;
					nearest_landmark = k;
				}
			}

			// If we found a landmark...
			if (nearest_landmark > -1) {
				// Calculate multivariate Gaussian distribution.
				float mu_x = map_landmarks.landmark_list[nearest_landmark].x_f;
				float mu_y = map_landmarks.landmark_list[nearest_landmark].y_f;
				double std_x = std_landmark[0];
				double std_y = std_landmark[1];
				double fraction_one = pow(obsT.x-mu_x, 2)/(2*std_x*std_x);
				double fraction_two = pow(obsT.y-mu_y, 2)/(2*std_y*std_y);
				long double multiplier = (1/(2*M_PI*std_x*std_y))*exp(-(fraction_one+fraction_two));

				// Adjust particle weight.
				particles[i].weight *= multiplier;

				// Update particle associations.
				associations.push_back(map_landmarks.landmark_list[nearest_landmark].id_i);
				sense_x.push_back(obsT.x);
				sense_y.push_back(obsT.y);
			}
		}

		// Record final particle weight.
		weights[i] = particles[i].weight;

		// Assign associations.
		this->SetAssociations(particles[i], associations, sense_x, sense_y);
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	std::vector<Particle> new_particles;
	std::discrete_distribution<> d(weights.begin(), weights.end());
	std::random_device rd;
    std::mt19937 gen(rd());
	for (int i = 0; i < num_particles; ++i) {
		new_particles.push_back(particles[d(gen)]);
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	// particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
