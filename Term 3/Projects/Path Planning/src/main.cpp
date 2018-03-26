#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

const int PATH_SIZE = 50;
const double HORIZON_LENGTH = 30.0; // meters
const double TIME_INC = .02; // seconds per car movement from one point to the next
const int LANE_WIDTH = 4; // meters
const int LANE_CENTER = 2; // meters
const double TARGET_VELOCITY = 49.5; // mph
const double VELOCITY_INC = .224; // mph = 0.1mps
const double MIN_VELOCITY = 10; // mph

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2) {
  return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}

int ClosestWaypoint(double x, double y, const vector<double> &maps_x,
        const vector<double> &maps_y) {
  double closestLen = 100000; //large number
  int closestWaypoint = 0;

  for (int i = 0; i < maps_x.size(); i++) {
    double map_x = maps_x[i];
    double map_y = maps_y[i];
    double dist = distance(x,y,map_x,map_y);
    if(dist < closestLen) {
      closestLen = dist;
      closestWaypoint = i;
    }
  }

  return closestWaypoint;
}

int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x,
        const vector<double> &maps_y) {
  int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

  double map_x = maps_x[closestWaypoint];
  double map_y = maps_y[closestWaypoint];

  double heading = atan2((map_y-y),(map_x-x));

  double angle = fabs(theta-heading);
  angle = min(2*pi() - angle, angle);

  if (angle > pi()/4) {
    closestWaypoint++;
    if (closestWaypoint == maps_x.size()) {
      closestWaypoint = 0;
    }
  }

  return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta,
        const vector<double> &maps_x, const vector<double> &maps_y) {
  int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

  int prev_wp;
  prev_wp = next_wp-1;
  if(next_wp == 0) {
    prev_wp  = maps_x.size()-1;
  }

  double n_x = maps_x[next_wp]-maps_x[prev_wp];
  double n_y = maps_y[next_wp]-maps_y[prev_wp];
  double x_x = x - maps_x[prev_wp];
  double x_y = y - maps_y[prev_wp];

  // find the projection of x onto n
  double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
  double proj_x = proj_norm*n_x;
  double proj_y = proj_norm*n_y;

  double frenet_d = distance(x_x,x_y,proj_x,proj_y);

  //see if d value is positive or negative by comparing it to a center point

  double center_x = 1000-maps_x[prev_wp];
  double center_y = 2000-maps_y[prev_wp];
  double centerToPos = distance(center_x,center_y,x_x,x_y);
  double centerToRef = distance(center_x,center_y,proj_x,proj_y);

  if(centerToPos <= centerToRef) {
    frenet_d *= -1;
  }

  // calculate s value
  double frenet_s = 0;
  for(int i = 0; i < prev_wp; i++) {
    frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
  }

  frenet_s += distance(0,0,proj_x,proj_y);

  return {frenet_s,frenet_d};
}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s,
        const vector<double> &maps_x, const vector<double> &maps_y) {
  int prev_wp = -1;

  while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1))) {
    prev_wp++;
  }

  int wp2 = (prev_wp+1)%maps_x.size();

  double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),
                          (maps_x[wp2]-maps_x[prev_wp]));
  // the x,y,s along the segment
  double seg_s = (s-maps_s[prev_wp]);

  double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
  double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

  double perp_heading = heading-pi()/2;

  double x = seg_x + d*cos(perp_heading);
  double y = seg_y + d*sin(perp_heading);

  return {x,y};
}

double convertMphToMps(double mph) {
  return mph/2.24; // rough conversion
}

// Creates a spline that interpolates a set of generated waypoints 30, 60, and
// 90 meters ahead of the car's current location.
tk::spline createSpline(const vector<double> previous_path_x,
        const vector<double> previous_path_y, double ref_x, double ref_y,
        double prev_ref_x, double prev_ref_y, double ref_yaw, double car_s,
        int current_lane, const vector<double> &maps_s,
        const vector<double> &maps_x, const vector<double> &maps_y) {

  int prev_path_size = previous_path_x.size();

  // First, create a list of widely spaced (x,y) waypoints, evenly spaced at 30m.
  // Later, we'll interpolate these waypoints with a spline and fill it with more
  // points at specified intervals to give the car the desired speed.
  vector<double> waypoints_x;
  vector<double> waypoints_y;

  // Begin with the 2 starting reference points.
  waypoints_x.push_back(prev_ref_x);
  waypoints_x.push_back(ref_x);
  waypoints_y.push_back(prev_ref_y);
  waypoints_y.push_back(ref_y);

  // In Frenet coordinates, add evenly spaced (30m apart) waypoints ahead of the car.
  double next_d = LANE_CENTER + LANE_WIDTH * current_lane;
  vector<double> waypoint_0 = getXY(car_s+30, next_d, maps_s, maps_x, maps_y);
  vector<double> waypoint_1 = getXY(car_s+60, next_d, maps_s, maps_x, maps_y);
  vector<double> waypoint_2 = getXY(car_s+90, next_d, maps_s, maps_x, maps_y);

  waypoints_x.push_back(waypoint_0[0]);
  waypoints_x.push_back(waypoint_1[0]);
  waypoints_x.push_back(waypoint_2[0]);
  waypoints_y.push_back(waypoint_0[1]);
  waypoints_y.push_back(waypoint_1[1]);
  waypoints_y.push_back(waypoint_2[1]);

  // Shift the waypoints reference angle to 0 degrees relative to the car,
  // to make the math easier later.
  for (int i = 0; i < waypoints_x.size(); i++) {
    double shift_x = waypoints_x[i] - ref_x;
    double shift_y = waypoints_y[i] - ref_y;
    waypoints_x[i] = shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw);
    waypoints_y[i] = shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw);
  }

  // Create a spline to interpolate the waypoints.
  tk::spline s;
  s.set_points(waypoints_x, waypoints_y);
  return s;
}

// Generates the actual (x,y) points we will use for the path planner.
void generatePath(const vector<double> previous_path_x,
        const vector<double> previous_path_y, tk::spline s, double ref_x,
        double ref_y, double ref_yaw, double ref_velocity,
        vector<double> &next_x_vals, vector<double> &next_y_vals) {

  int prev_path_size = previous_path_x.size();

  // Start the path with all the points leftover from the previous path.
  for (int i = 0; i < prev_path_size; i++) {
    next_x_vals.push_back(previous_path_x[i]);
    next_y_vals.push_back(previous_path_y[i]);
  }

  // Calculate how to break up the spline points so the car travels at the
  // desired reference velocity. We do this by predicting a horizon 30m into
  // the future, and linearly seperating the spline into sections. The length
  // of each section is determined by the reference velocity.
  double horizon_x = HORIZON_LENGTH;
  double horizon_y = s(horizon_x);
  double horizon_dist = sqrt((horizon_x * horizon_x) + (horizon_y * horizon_y));
  double N_sections = horizon_dist / (TIME_INC * convertMphToMps(ref_velocity));
  double section_start_x = 0;

  // Fill up the rest of the path!
  for (int i = 0; i < PATH_SIZE - prev_path_size; i++) {
    double section_x = section_start_x + (horizon_x / N_sections);
    double section_y = s(section_x);

    section_start_x = section_x;

    // Rotate the point back to the reference orientation to undo
    // the earlier rotation and get the point in global coordinates.
    double x_global = section_x * cos(ref_yaw) - section_y * sin(ref_yaw);
    double y_global = section_x * sin(ref_yaw) + section_y * cos(ref_yaw);

    x_global += ref_x;
    y_global += ref_y;

    next_x_vals.push_back(x_global);
    next_y_vals.push_back(y_global);
  }
}

// Check if there is a car in front, behind, to the left, or to the right.
void check_car_surroundings(const vector<vector<double>> sensor_fusion,
        const int prev_path_size, double car_s, const int current_lane,
        bool &car_front, bool &car_back, bool &car_left, bool &car_right) {
  for (int i = 0; i < sensor_fusion.size(); i++) {
    double other_car_d = sensor_fusion[i][6];
    double vx = sensor_fusion[i][3];
    double vy = sensor_fusion[i][4];
    double speed = sqrt(vx*vx + vy*vy);
    double other_car_s = sensor_fusion[i][5];
    // Project the s value ahead into the future using the previous path.
    other_car_s += ((double)prev_path_size*TIME_INC*speed);
    if (other_car_d > (LANE_WIDTH * current_lane) && 
            other_car_d < (LANE_WIDTH * current_lane + LANE_WIDTH)) {
      // The other car is in my lane!
      if (other_car_s > car_s && other_car_s - car_s < 30) {
        // Other car is less than 30 meters in front of me!
        car_front = true;
      } else if (other_car_s < car_s && car_s - other_car_s < 5) {
        // Other car is less than 5 meters behind me!
        car_back = true;
      }
    } else if (other_car_d > (LANE_WIDTH * (current_lane - 1)) && 
            other_car_d < (LANE_WIDTH * (current_lane - 1) + LANE_WIDTH)) {
      // The other car is in the lane to the left of me!
      if ((other_car_s > car_s && other_car_s - car_s < 30) || 
              (other_car_s <= car_s && car_s - other_car_s < 2)) {
        // Other car is less than 30 meters in front of me, or less than 2
        // meters behind me!
        car_left = true;
      }
    } else if (other_car_d > (LANE_WIDTH * (current_lane + 1)) && 
            other_car_d < (LANE_WIDTH * (current_lane + 1) + LANE_WIDTH)) {
      // The other car is in the lane to the right of me!
      if ((other_car_s > car_s && other_car_s - car_s < 30) || 
              (other_car_s <= car_s && car_s - other_car_s < 2)) {
        // Other car is less than 30 meters in front of me, or less than 2
        // meters behind me!
        car_right = true;
      }
    }
  }
}

// Check if there is a car less than dist meters in front of me in the given lane.
bool check_car_in_lane(const vector<vector<double>> sensor_fusion,
        const int prev_path_size, double car_s, const int lane, const int dist) {
  for (int i = 0; i < sensor_fusion.size(); i++) {
    double other_car_d = sensor_fusion[i][6];
    if (other_car_d > (LANE_WIDTH * lane) && 
            other_car_d < (LANE_WIDTH * lane + LANE_WIDTH)) {
      // The other car is in my lane!
      double vx = sensor_fusion[i][3];
      double vy = sensor_fusion[i][4];
      double speed = sqrt(vx*vx + vy*vy);
      double other_car_s = sensor_fusion[i][5];
      // Project the s value ahead into the future using the previous path.
      other_car_s += ((double)prev_path_size*TIME_INC*speed);
      if (other_car_s > car_s && other_car_s - car_s < dist) {
        // Other car is less than dist meters in front of me!
        return true;
      }
    }
  }
  return false;
}

enum Behavior {
  FORWARD,
  PREP_CHANGE_LANES_LEFT,
  PREP_CHANGE_LANES_RIGHT,
  CHANGE_LANES_LEFT,
  CHANGE_LANES_RIGHT
};

// Use subsumption architecture to ensure the car avoids collisions.
void check_behavior(const vector<vector<double>> sensor_fusion,
        const int prev_path_size, double car_s, double car_d, int &current_lane,
        double &ref_velocity, Behavior &behavior) {
          
  bool car_front = false;
  bool car_back = false;
  bool car_left = false;
  bool car_right = false; 
  
  check_car_surroundings(sensor_fusion, prev_path_size, car_s, current_lane,
          car_front, car_back, car_left, car_right);

  double current_center = LANE_CENTER + LANE_WIDTH * current_lane;
  
  switch(behavior) {
    case FORWARD:
      if (car_front) {
        // There is a car in front of me!
        if (current_lane > 0 && !car_left) {
          // Left lane looks open.
          bool car_ahead_left = check_car_in_lane(sensor_fusion,
                  prev_path_size, car_s, current_lane-1, 50);
          if (car_right || !car_ahead_left || current_lane == 2) {
            // Can't change right, or left lane is clear, so let's change left.
            cout << "Prep left lane change!\n";
            behavior = PREP_CHANGE_LANES_LEFT;
          } else {
            // Left lane is not clear up ahead, so let's change right.
            cout << "Prep right lane change!\n";
            behavior = PREP_CHANGE_LANES_RIGHT;
          }
        } else if (current_lane < 2 && !car_right) {
          // Right lane looks open.
          bool car_ahead_right = check_car_in_lane(sensor_fusion,
                  prev_path_size, car_s, current_lane+1, 50);
          if (car_left || !car_ahead_right || current_lane == 0) {
            // Can't change left, or right lane is clear, so let's change right.
            cout << "Prep right lane change!\n";
            behavior = PREP_CHANGE_LANES_RIGHT;
          } else {
            // Right lane is not clear up ahead, so let's change left.
            cout << "Prep left lane change!\n";
            behavior = PREP_CHANGE_LANES_LEFT;
          }
        } else {
          // Lanes on both sides are blocked. Just slow down so we don't hit 
          // the car in front.
          if (!car_back && ref_velocity > MIN_VELOCITY) {
            ref_velocity -= VELOCITY_INC;
          }
          // Now check the other lanes to see if we can move further away.
          if (current_lane == 0) {
            // Check the right-most lane.
            bool car_in_right_lane = check_car_in_lane(sensor_fusion,
                    prev_path_size, car_s, 2, 30);
            if (car_in_right_lane) {
              cout << "Prep right lane change!\n";
              behavior = PREP_CHANGE_LANES_RIGHT;
            }
          } else if (current_lane == 2) {
            // Check the left-most lane.
            bool car_in_left_lane = check_car_in_lane(sensor_fusion,
                    prev_path_size, car_s, 0, 30);
            if (car_in_left_lane) {
              cout << "Prep left lane change!\n";
              behavior = PREP_CHANGE_LANES_LEFT;
            }
          }
        }
      } else if (ref_velocity < TARGET_VELOCITY) {
        // Speed up to the target velocity. Go forward!
        ref_velocity += VELOCITY_INC;
      }
      break;
    case PREP_CHANGE_LANES_LEFT:
      if (car_left) {
        // There's a car to the left! We can't change lanes yet.
        if (car_back && !car_front) {
          // There's a car behind us!
          if (ref_velocity < TARGET_VELOCITY) {
            // Speed up so we don't kill the guy behind us.
            ref_velocity += VELOCITY_INC;
          }
        } else {
          // Slow down until it's safe to change lanes.
          if (ref_velocity > MIN_VELOCITY) {
            ref_velocity -= VELOCITY_INC;
          }
        }
      } else {
        // It's safe! Do the lane change.
        cout << "Changing lanes to the left!\n";
        current_lane--;
        behavior = CHANGE_LANES_LEFT;
      }
      break;
    case PREP_CHANGE_LANES_RIGHT:
      if (car_right) {
        // There's a car to the right! We can't change lanes yet.
        if (car_back && !car_front) {
          // There's a car behind us!
          if (ref_velocity < TARGET_VELOCITY) {
            // Speed up so we don't kill the guy behind us.
            ref_velocity += VELOCITY_INC;
          }
        } else {
          // Slow down until it's safe to change lanes.
          if (ref_velocity > MIN_VELOCITY) {
            ref_velocity -= VELOCITY_INC;
          }
        }
      } else {
        // It's safe! Do the lane change.
        cout << "Changing lanes to the right!\n";
        current_lane++;
        behavior = CHANGE_LANES_RIGHT;
      }
      break;
    case CHANGE_LANES_LEFT:
      if (abs(car_d - current_center) < 0.1 /* meters */) {
        // I'm at the center of the new lane! Continue moving forward.
        cout << "Going forward!\n";
        behavior = FORWARD;
      }
      if (car_front) {
        // There is a car in front of me!
        // Slow down so we don't hit the car in front.
        if (ref_velocity > MIN_VELOCITY) {
          ref_velocity -= VELOCITY_INC;
        }
      } else if (ref_velocity < TARGET_VELOCITY) {
        // Speed up to the target velocity.
        ref_velocity += VELOCITY_INC;
      }
      break;
    case CHANGE_LANES_RIGHT:
      if (abs(car_d - current_center) < 0.1 /* meters */) {
        // I'm at the center of the new lane! Continue moving forward.
        cout << "Going forward!\n";
        behavior = FORWARD;
      }
      if (car_front) {
        // There is a car in front of me!
        // Slow down so we don't hit the car in front.
        if (ref_velocity > MIN_VELOCITY) {
          ref_velocity -= VELOCITY_INC;
        }
      } else if (ref_velocity < TARGET_VELOCITY) {
        // Speed up to the target velocity.
        ref_velocity += VELOCITY_INC;
      }
      break;
  }
}

// Parses the localization data being sent over uWebSockets.
json parseLocalizationData(char *data, size_t length, bool &manualMode) {
  // "42" at the start of the message means there's a websocket message event.
  // The 4 signifies a websocket message
  // The 2 signifies a websocket event
  //auto sdata = string(data).substr(0, length);
  //cout << sdata << endl;
  if (length && length > 2 && data[0] == '4' && data[1] == '2') {

    auto s = hasData(data);

    if (s != "") {
      json j = json::parse(s);
      
      string event = j[0].get<string>();

      if (event == "telemetry") {
        return j;
      }
    } else {
      manualMode = true;
    }
  }
  return json::array();
}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  // Start in lane 1
  int current_lane = 1;

  // The car's motion will target this velocity.
  double ref_velocity = 0; // mph

  // This is the starting/default behavior of the car (go forward).
  Behavior behavior = FORWARD;

  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
                &map_waypoints_dx,&map_waypoints_dy,&current_lane,
                &ref_velocity,&behavior](uWS::WebSocket<uWS::SERVER> ws, 
                char *data, size_t length, uWS::OpCode opCode) {

    bool manualMode = false;
    json j = parseLocalizationData(data, length, manualMode);

    if (!j.empty()) {
      // Main car's localization Data
      double car_x = j[1]["x"];
      double car_y = j[1]["y"];
      double car_s = j[1]["s"];
      double car_d = j[1]["d"];
      double car_yaw = j[1]["yaw"];
      double car_speed = j[1]["speed"];

      // Previous path data given to the Planner
      vector<double> previous_path_x = j[1]["previous_path_x"];
      vector<double> previous_path_y = j[1]["previous_path_y"];
      // Previous path's end s and d values 
      double end_path_s = j[1]["end_path_s"];
      double end_path_d = j[1]["end_path_d"];

      // Sensor Fusion Data, a list of all other cars on the same side of the road.
      vector<vector<double>> sensor_fusion = j[1]["sensor_fusion"];

      int prev_path_size = previous_path_x.size();

      if (prev_path_size > 0) {
        car_s = end_path_s;
      }

      // Use subsumption architecture to ensure the car avoids collisions.
      check_behavior(sensor_fusion, prev_path_size, car_s, car_d, current_lane,
              ref_velocity, behavior);

      // We will build the path starting from reference x,y,yaw states. We will
      // either reference the car's current state, or the end state of the
      // previous path.
      double ref_x;
      double ref_y;
      double ref_yaw;
      double prev_ref_x;
      double prev_ref_y;

      if (prev_path_size < 2) {
        // If previous path size is not big enough, use the car's current state.
        ref_x = car_x;
        ref_y = car_y;
        ref_yaw = deg2rad(car_yaw);
        prev_ref_x = car_x - cos(car_yaw);
        prev_ref_y = car_y - sin(car_yaw);
      } else {
        // Otherwise, use the previous path's end point as starting reference.
        ref_x = previous_path_x[prev_path_size-1];
        ref_y = previous_path_y[prev_path_size-1];
        prev_ref_x = previous_path_x[prev_path_size-2];
        prev_ref_y = previous_path_y[prev_path_size-2];
        ref_yaw = atan2(ref_y - prev_ref_y, ref_x - prev_ref_x);
      }

      // Generate a spline that interpolates a set of waypoints, as well as the
      // reference yaw of the car.
      tk::spline spline = createSpline(previous_path_x, previous_path_y,
              ref_x, ref_y, prev_ref_x, prev_ref_y, ref_yaw, car_s,
              current_lane, map_waypoints_s, map_waypoints_x,
              map_waypoints_y);

      json msgJson;

      // Define the actual (x,y) points we will use for the path planner.
      vector<double> next_x_vals;
      vector<double> next_y_vals;

      generatePath(previous_path_x, previous_path_y, spline, ref_x, ref_y,
              ref_yaw, ref_velocity, next_x_vals, next_y_vals);
      
      msgJson["next_x"] = next_x_vals;
      msgJson["next_y"] = next_y_vals;
      auto msg = "42[\"control\","+ msgJson.dump()+"]";
      //this_thread::sleep_for(chrono::milliseconds(1000));
      ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
    }
    if (manualMode) {
      // Manual driving
      std::string msg = "42[\"manual\",{}]";
      ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
