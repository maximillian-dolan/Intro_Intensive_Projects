// This code uses various numerical integral methods to show that
// the integral of a Gaussian curve is roughly equal to the square
// root of pi

// Include Headers
#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <omp.h>

// Define midpoint method function
double midpoint_integral_func(std::vector<double> x_values, double x_gap)
{
    double midpoint_integral = 0;
    std::vector<double> midpoint_y_values;

    // Calculate y midpoints
    for (int i = 0 ; i<x_values.size() ; i++)
    {
        midpoint_y_values.push_back(std::exp(-std::pow((x_values[i]+x_gap/2),2)));
    }

    // Add area of each rectangle together
    for (double i : midpoint_y_values)
    {
        midpoint_integral += i*x_gap;
    }

    return midpoint_integral;
}

// Define trapezium rule
double trapezium_integral_func(std::vector<double> x_values, double x_gap)
{
    double trapezium_integral = 0;
    std::vector<double> trapezium_y_values;

    // Calculate y values
    for (double x : x_values)
    {
        trapezium_y_values.push_back(std::exp(-std::pow(x,2)));
    }

    // Add area of each rectangle and the triangle on top of it together
    for (int i = 0 ; i<trapezium_y_values.size() ; i++)
    {
        trapezium_integral += trapezium_y_values[i]*x_gap;
        trapezium_integral += (trapezium_y_values[i+1]-trapezium_y_values[i])*x_gap/2;
    }

    return trapezium_integral;
}

// Define simpsons rule
double simpsons_integral_func(std::vector<double> x_values, double x_gap)
{
    double simpsons_integral = 0;
    double x;

    // Perform Simpsons rule every two x points and add to integral
    for (int i = 0 ; i<(x_values.size()/2) ; i++)
    {
        x = x_values[2*i];
        simpsons_integral += x_gap * (std::exp(-std::pow(x,2)) + 4*std::exp(-std::pow((x+x_gap),2)) + std::exp(-std::pow((x+2*x_gap),2))) / 3;
    }

    return simpsons_integral;
}

// Define function for making x_values array
std::vector<double> x_values_generator(double x_gap)
{
    // Initialise array
    std::vector<double> x_values;
    x_values.push_back(-5);

    // Loop to add next value in array 
    for (int i = 0 ; x_values[i]<=5 ; i++)
    {
        x_values.push_back(x_values[i]+x_gap);
    }

    return x_values;
}

// Calculates minimum x_gap value for 1e-9 accuracy
double minimum_x_gap(double (*func)(std::vector<double>, double ))
{
    double integral1 = 1;
    double integral2;
    double x_gap1 = 1;
    double x_gap2;
    std::vector<double> x_values1;
    std::vector<double> x_values2;

    // When a x_gap value is found that meets the level of accuracy,
    // it must also show that the value 10 iterations less than that
    // also does so. Else found value might be a chance value that
    // meets the criteria and not a proper convergence.
    while ((fabs(integral1 - 1.772453851) >= 1e-9) or (fabs(integral2 - 1.772453851) >= 1e-9))
    {   
        x_gap2 = x_gap1 - 0.00001;
        x_values2 = x_values_generator(x_gap2);
        integral2 = func(x_values2, x_gap2);
        x_gap1 -= 0.000001;
        x_values1 = x_values_generator(x_gap1);
        integral1 = func(x_values1, x_gap1);     
    }

    // Return value
    return x_gap1;
}


int main()
{
    // Define x_values array and the x_gap
    double x_gap = 0.1;
    auto x_values = x_values_generator(x_gap);

    // Calculate integrals
    double midpoint_integral;
    double trapezium_integral;
    double simpsons_integral;

    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                midpoint_integral = midpoint_integral_func(x_values, x_gap);
            }
            #pragma omp section
            {
                trapezium_integral = trapezium_integral_func(x_values, x_gap);
            }
            #pragma omp section
            {
                simpsons_integral = simpsons_integral_func(x_values, x_gap);
            }
        }
    }

    // Set precision for cout
    std::cout << std::setprecision(16);

    // Print results for gap of 0.1
    std::cout << "-----" << std::endl;
    std::cout << "For an x spacing of 0.1:" << std::endl;
    std::cout << "The midpoint integral is  " << midpoint_integral  << std::endl;
    std::cout << "The trapezium integral is " << trapezium_integral  << std::endl;
    std::cout << "The simpsons integral is  " << simpsons_integral  << std::endl;
    std::cout << "-----" << std::endl;

    // Find out what x_gap needed to get accuracy to 9 decimal points
    double midpoint_min;
    double trapezium_min;
    double simpsons_min;

    // Run programs in parallel as they all take a while and independant of one another
    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                midpoint_min = minimum_x_gap(midpoint_integral_func);
            }
            #pragma omp section
            {
                trapezium_min = minimum_x_gap(trapezium_integral_func);
            }
            #pragma omp section
            {
                simpsons_min = minimum_x_gap(simpsons_integral_func);
            }
        }
    }

    // Print min x_gap for each method
    std::cout << std::setprecision(8);
    std::cout << "The minimum x_gap needed for 1e-9 level of accuracy:" << std::endl;
    std::cout << "Midpoint:  " << midpoint_min << std::endl;
    std::cout << "Trapezium: " << trapezium_min << std::endl;
    std::cout << "Simpsons:  " << simpsons_min << std::endl;

    std::cout << "-----" << std::endl;

    return EXIT_SUCCESS;
}