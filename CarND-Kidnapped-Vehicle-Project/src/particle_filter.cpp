/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"
#include "particle_filter.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  std::default_random_engine gen;
  if (is_initialized)
  {
    return;
  }

  num_particles = 40; // TODO: Set the number of particles
  double stdx = std[0];
  double stdy = std[1];
  double stdTheta = std[2];

  // Distribucion normal

  std::normal_distribution<double> dist_x(x, stdx);
  std::normal_distribution<double> dist_y(y, stdy);
  std::normal_distribution<double> dist_theta(theta, stdTheta);

  // Generar particulas{
  for (int i = 0; i < num_particles; i++)
  {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;

    particles.push_back(particle);
    weights.push_back(particle.weight);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  std::default_random_engine gen;
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  double stdx = std_pos[0];
  double stdy = std_pos[1];
  double stdTheta = std_pos[2];

  // distribuciones

  for (int i = 0; i < num_particles; i++)
  {
    double particle_X = particles[i].x;
    double particle_y = particles[i].y;
    double particle_theta = particles[i].theta;

    double predx;
    double predy;
    double predTheta;
    if (fabs(yaw_rate) < 0.001)
    {
      predx = particle_X + velocity * delta_t * cos(particle_theta);
      predy = particle_y + velocity * delta_t * sin(particle_theta);
      predTheta = particle_theta;
    }
    else
    {
      predx = particle_X + (velocity / yaw_rate) * (sin(particle_theta + (yaw_rate * delta_t)) - sin(particle_theta));
      predTheta = particle_theta + (yaw_rate * delta_t);
      predy = particle_y + (velocity / yaw_rate) * (cos(particle_theta) - cos(particle_theta + (yaw_rate * delta_t)));
    }

    std::normal_distribution<double> dist_x(predx, stdx);
    std::normal_distribution<double> dist_y(predy, stdy);
    std::normal_distribution<double> dist_theta(predTheta, stdTheta);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for (unsigned int i = 0; i < observations.size(); i++)
  {
    double min_distant = 10000;

    int mapId = -1;
    for (unsigned int j = 0; j < predicted.size(); j++)
    {
      int pred_id = predicted[j].id;
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (distance < min_distant)
      {
        min_distant = distance;
        mapId = pred_id;
      }
    }
    observations[i].id = mapId;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  int i, j;
  /*This variable is used for normalizing weights of all particles and bring them in the range
    of [0, 1]*/
  double weight_normalizer = 0.0;

  for (i = 0; i < num_particles; i++)
  {
    double particle_x = particles[i].x;
    double particle_y = particles[i].y;
    double particle_theta = particles[i].theta;

    /*Step 1: Transform observations from vehicle co-ordinates to map co-ordinates.*/
    //Vector containing observations transformed to map co-ordinates w.r.t. current particle.
    vector<LandmarkObs> transformed_observations;

    //Transform observations from vehicle's co-ordinates to map co-ordinates.
    for (j = 0; j < observations.size(); j++)
    {
      LandmarkObs transformed_obs;
      transformed_obs.id = j;
      transformed_obs.x = particle_x + (cos(particle_theta) * observations[j].x) - (sin(particle_theta) * observations[j].y);
      transformed_obs.y = particle_y + (sin(particle_theta) * observations[j].x) + (cos(particle_theta) * observations[j].y);
      transformed_observations.push_back(transformed_obs);
    }

    /*Step 2: Filter map landmarks to keep only those which are in the sensor_range of current 
     particle. Push them to predictions vector.*/
    vector<LandmarkObs> predicted_landmarks;
    for (j = 0; j < map_landmarks.landmark_list.size(); j++)
    {
      Map::single_landmark_s current_landmark = map_landmarks.landmark_list[j];
      if ((fabs((particle_x - current_landmark.x_f)) <= sensor_range) && (fabs((particle_y - current_landmark.y_f)) <= sensor_range))
      {
        predicted_landmarks.push_back(LandmarkObs{current_landmark.id_i, current_landmark.x_f, current_landmark.y_f});
      }
    }

    /*Step 3: Associate observations to lpredicted andmarks using nearest neighbor algorithm.*/
    //Associate observations with predicted landmarks
    dataAssociation(predicted_landmarks, transformed_observations);

    /*Step 4: Calculate the weight of each particle using Multivariate Gaussian distribution.*/
    //Reset the weight of particle to 1.0
    particles[i].weight = 1.0;

    double sigma_x = std_landmark[0];
    double sigma_y = std_landmark[1];
    double sigma_x_2 = pow(sigma_x, 2);
    double sigma_y_2 = pow(sigma_y, 2);
    double normalizer = (1.0 / (2.0 * M_PI * sigma_x * sigma_y));
    int k, l;

    /*Calculate the weight of particle based on the multivariate Gaussian probability function*/
    for (k = 0; k < transformed_observations.size(); k++)
    {
      double trans_obs_x = transformed_observations[k].x;
      double trans_obs_y = transformed_observations[k].y;
      double trans_obs_id = transformed_observations[k].id;
      double multi_prob = 1.0;

      for (l = 0; l < predicted_landmarks.size(); l++)
      {
        double pred_landmark_x = predicted_landmarks[l].x;
        double pred_landmark_y = predicted_landmarks[l].y;
        double pred_landmark_id = predicted_landmarks[l].id;

        if (trans_obs_id == pred_landmark_id)
        {
          multi_prob = normalizer * exp(-1.0 * ((pow((trans_obs_x - pred_landmark_x), 2) / (2.0 * sigma_x_2)) + (pow((trans_obs_y - pred_landmark_y), 2) / (2.0 * sigma_y_2))));
          particles[i].weight *= multi_prob;
        }
      }
    }
    weight_normalizer += particles[i].weight;
  }

  /*Step 5: Normalize the weights of all particles since resmapling using probabilistic approach.*/
  for (int i = 0; i < particles.size(); i++)
  {
    particles[i].weight /= weight_normalizer;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample()
{
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  std::default_random_engine gen;

  // Get weights and max weight.
  vector<double> weights;
  double maxWeight = -10000;
  for (int i = 0; i < num_particles; i++)
  {
    weights.push_back(particles[i].weight);
    if (particles[i].weight > maxWeight)
    {
      maxWeight = particles[i].weight;
    }
  }

  // Creating distributions.
  std::uniform_real_distribution<double> distDouble(0.0, maxWeight);
  std::uniform_int_distribution<int> distInt(0, num_particles - 1);

  // Generating index.
  int index = distInt(gen);

  double beta = 0.0;

  // the wheel
  vector<Particle> resampledParticles;
  for (int i = 0; i < num_particles; i++)
  {
    beta += distDouble(gen) * 2.0;
    while (beta > weights[index])
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampledParticles.push_back(particles[index]);
  }

  particles = resampledParticles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}