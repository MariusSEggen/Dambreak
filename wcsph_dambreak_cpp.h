#ifndef WCSPH_DAMBREAK_H
#define WCSPH_DAMBREAK_H

#define _XOPEN_SOURCE 500

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <ctime>
#include <unordered_map>

typedef double real_t;
#define REAL_FMT "%.4e"
typedef int64_t int_t;
#define INT_FMT "%ld"

// Parameters
#define A (1.0)
// Dam height
#define T (0.6 * A + 3*DELTA)
// Dam width
#define L (1.2 * A + 5*DELTA)
// Tank width
#define B (3.22 * A)
// Spatial resolution
#define DELTA (0.01)

// #define H (0.94*DELTA*sqrt(2.0))
// Hardwire sqrt(2) to avoid recomputation of constant value
#define H (0.94*DELTA*1.4142135623)

// Constants derived from parameters
const real_t
    D_tank  = 1.0 * A,
    scale_k = 3.0,
    density = 1e3,
    dt = 1e-4,          // Time step
    sos = 50.0;         // Acoustic wave prop. (speed of sound in medium)

const int_t
    free_surface = 30;

// Particle data structure
typedef struct particle_t{
    // Particle information
    real_t
        x[2],   // 2D position
        v[2],   // 2D velocity
        mass,   // Mass
        rho,    // Density
        p,      // Pressure
        type,   // Type of particle
        hsml;   // Related to resolution (not fully understood yet)
       
    int_t
        interactions,
        cell_x,
        cell_y,
        id;

    // Differentials
    real_t
        indvxdt[2],
        exdvxdt[2],
        dvx[2],
        drhodt;
    // Density correction
    real_t
        avrho,
        w_sum;
} particle_t;

// Particle list indices, to separate variables from memory structure
#define X(k)     p_list[(k)].x[0]
#define Y(k)     p_list[(k)].x[1]
#define VX(k)    p_list[(k)].v[0]
#define VY(k)    p_list[(k)].v[1]
#define M(k)     p_list[(k)].mass
#define RHO(k)   p_list[(k)].rho
#define P(k)     p_list[(k)].p
#define TYPE(k)  p_list[(k)].type
#define HSML(k)  p_list[(k)].hsml
#define INTER(k) p_list[(k)].interactions

#define INDVXDT(k,i) p_list[(k)].indvxdt[(i)]
#define EXDVXDT(k,i) p_list[(k)].exdvxdt[(i)]
#define DVX(k,i) p_list[(k)].dvx[(i)]
#define DRHODT(k) p_list[(k)].drhodt

#define AVRHO(k) p_list[(k)].avrho
#define WSUM(k) p_list[(k)].w_sum


// Pairwise interaction
typedef struct {
    int_t i, j;     // Which particles interact?
    real_t
        r,          // Distance between particles (Euclid)
        q,          // Distance normalized to H (resolution)
        w,          // TODO: consult paper for meaning of this
        dwdx[2];    // Influence on velocity
} pair_t;


// Allocate and initialize particle list according to parameters
void initialize ( void );
// Destroy particle list (changed during computation)
void finalize ( void );

// Computation
void find_neighbors ( void );               // Build list of close pairs
void kernel ( void );                       // Pairwise interactions
void int_force ( void );                    // Internal force (interactions)
void cont_density ( void );                 // 
void ext_force ( void );                    // External force (gravity)
void correction ( void );                   // Correction term for density
void time_step ( void );                    // Dist. properties in time step
void time_integration ( void );             // Loop, propagation of properties
int main ( int argc, char **argv );         // Entry point
void resize_particle_list ( void );         // Realloc. list when full
void resize_pair_list ( void );             // Realloc. pairs when full
void generate_virtual_particles ( void );   // Create ghosts at boundary

// State I/O
void write_positions ( FILE *out, int_t start, int_t end );
void write_state ( FILE *out, int_t start, int_t end );
#endif
