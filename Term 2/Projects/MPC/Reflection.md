# MPC (Model Predictive Control) Project
## by Soeren Walls

### The Model

My model of the car follows the Udacity MPC algorithm quite closely.

The state vector contains information about the state of the car, including:

- x position
- y position
- steering angle (psi)
- velocity
- cross track error
- angle (psi) error

The actuators on the car are the steering wheel and the throttle. As such, my MPC controller can send signals to either one of these main controls in order to alter the angle the car is facing, and its acceleration.

In order to update the polynomial that models the predicted behavior of the car, I use the classic MPC approach of treating the problem like an optimization problem, where the optimal path is the center of the lane. This is accomplished using a cost function, where the cost is a combination of different attributes, such as:

- cross track error
- angle (psi) error
- actuator changes

### Timestep Length and Elapsed Duration

There are a few parameters that come into play here. As a reminder from the MPC lesson:

*T* = the prediction horizon (the duration over which future predictions are made)

*N* is the number of timesteps in the horizon. *dt* is how much time elapses between actuations. For example, if *N* were 20 and *dt* were 0.5, then *T* would be 10 seconds.

*T* should be as large as possible, while *dt* should be as small as possible.

In my model, I attempted to stay true to this relationship by tuning these hyperparameters to try and get the best performance for the least amount of computing possible. In the end, I set *N*=10 and *dt*=0.1. This means my prediction horizon (*T*) is 1 second. While *T* could ideally be longer, 1 second is long enough for the state of a car to change significantly, especially when it's going 50+ mph, and so keeping a longer prediction horizon than that is simply unnecessary. Also, *dt*=0.1s is small enough that the MPC is able to respond quickly and accurately to error, but not so small that the computation takes longer than a single timestep to compute.

Before coming to this, I tried other values, like:
1. *N*=5 and *dt*=0.05 (*T* = 0.25)
2. *N*=20 and *dt*=0.5 (*T* = 4)
3. *N*=15 and *dt*=0.3 (*T* = 4.5)

But these were either too computationally expensive (2, 3) or did not provide a long enough prediction horizon (1). This is why I think the values I chose are ideal.

### Polynomial Fitting and MPC Preprocessing

Before actually calling `MPC.Solve()`, it is vital to preprocess the waypoints. Or at least, doing so makes future calculations a lot simpler.

Upon receiving a new state vector from the simulator, before doing any MPC calculations, I first shift the car's reference angle to 90 degrees (0 radians). This is because the car's initial frame of reference is in the positive y direction, which means the path ahead of it is vertical. Trying to make a polynomial along the y axis can be difficult, since it requires manipulating the coordinate system such that a function can have multiple y coordinates for a single value of x. Rather than manipulate the coordinate system, I alter the frame of reference by transforming it to the origin and rotating it to make a horizontally-oriented polynomial. This makes it much simpler to do calculations down the road (pun intended).

### Model Predictive Control with Latency

This project requires that the MPC algorithm should still work with 100ms of latency between actuator commands and actuator outputs, in order to more accurately model how a real car will behave. In my algorithm, I do this by altering the start state vector every time we receive new state data from the simulator. Specifically, I use the same transition functions from the MPC algorithm to simulate the car moving with the current steering angle and throttel value for 100ms from the start state. At the end of the simulation, I record the new predicted state and use this instead as the start state for the MPC algorithm. This allows the model to adapt to latency issues quite well, and perform almost as if there was no latency at all, with only slightly more computation required.