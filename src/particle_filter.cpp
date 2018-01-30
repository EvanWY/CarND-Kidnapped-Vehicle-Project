/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <iostream>
#include <iterator>
#include <math.h>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // create Gaussian distribution for x, y and theta.
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    num_particles = 1000;

    for (int id = 0; id < num_particles; ++id) {
        Particle p;
        p.id = id;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1;

        particles.push_back(p);
        weights.push_back(1);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    if (abs(yaw_rate) < 0.00001) {
        for (int i = 0; i < num_particles; ++i) {
            Particle& p = particles[i];
            p.x += (velocity * delta_t) * cos(p.theta) + dist_x(gen);
            p.y += (velocity * delta_t) * sin(p.theta) + dist_y(gen);
            p.theta += yaw_rate * delta_t + dist_theta(gen);
        }
    } else {
        for (int i = 0; i < num_particles; ++i) {
            Particle& p = particles[i];
            p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) + dist_x(gen);
            p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) + dist_y(gen);
            p.theta += yaw_rate * delta_t + dist_theta(gen);
        }
    }
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs>& predicted, const std::vector<LandmarkObs>& observations, Particle& particle)
{
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    particle.associations.clear();
    for (auto ob : observations){
        int cidx = 0;
        double obx = ob.x;
        double oby = ob.y;

        double cdist = dist(predicted[cidx].x, predicted[cidx].y, obx, oby);
        for (int i=1; i<predicted.size(); i++) {
            double d = dist(predicted[cidx].x, predicted[cidx].y, obx, oby);
            if (d < cdist) {
                cdist = d;
                cidx = i;
            }
        }

        particle.associations.push_back(cidx);
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
    const std::vector<LandmarkObs>& observations, const Map& map_landmarks)
{
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    for (int i = 0; i < num_particles; ++i) {
        Particle& p = particles[i];

        double theta = p.theta;
        double x = p.x;
        double y = p.y;

        std::vector<LandmarkObs> predicted; 
        for (auto lm : map_landmarks.landmark_list) {
            LandmarkObs predict_ob;
            predict_ob.id = lm.id_i;
            predict_ob.x = lm.x_f * cos(theta) + lm.y_f * sin(theta) + (lm.x_f - x);
            predict_ob.y = -lm.x_f * sin(theta) + lm.y_f * cos(theta) + (lm.y_f - y);
            predicted.push_back(predict_ob);
        }

        dataAssociation(predicted, observations, p);

        //double total_prob = p.weight;
        double total_prob = 1;

        for (int obid = 0; obid < observations.size(); obid++) {
            auto ob = observations[obid];
            auto pr = predicted[p.associations[obid]];
            double dx = ob.x - pr.x;
            double dy = ob.y - pr.y;

            double std = std_landmark[0];
            double px = 1.0/(std * 2.50662827463) * exp(- dx*dx /(2*std*std));
            std = std_landmark[1];
            double py = 1.0/(std * 2.50662827463) * exp(- dy*dy /(2*std*std));

            total_prob *= px * py;
        }

        p.weight = total_prob;
        weights[i] = total_prob;
    }
}

void ParticleFilter::resample()
{
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution


    double sum = accumulate(weights.begin(), weights.end(), 0);
    vector<Particle> newParticles;
    for (int i=0; i<num_particles; i++) {
        double val = (double)rand()/(double)RAND_MAX * sum;
        int idx = 0;
        val -= weights[idx];
        while (val > 0 && idx < weights.size()-1) {
            idx ++;
            val -= weights[idx];
        }

        newParticles.push_back(particles[idx]);
    }

    particles = newParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
    const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
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
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}
