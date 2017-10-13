// Set the timestep length and duration
const size_t N = 10;
const double dt = .1;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// How much attention should the cost function pay to each attribute?
// High weight = attribute has more cost (big effect)
// Low weight = attribute has less cost (small effect)
const int weight_cte = 6000;
const int weight_epsi = 6000;
const int weight_v = 1;
const int weight_delta = 5;
const int weight_a = 20;
const int weight_delta_gap = 200;
const int weight_a_gap = 10;

const double ref_cte = 0;
const double ref_epsi = 0;
const double ref_v = 100;

const size_t x_start = 0;
const size_t y_start = x_start + N;
const size_t psi_start = y_start + N;
const size_t v_start = psi_start + N;
const size_t cte_start = v_start + N;
const size_t epsi_start = cte_start + N;
const size_t delta_start = epsi_start + N;
const size_t a_start = delta_start + N - 1;