#include "wcsph_dambreak_cpp.h"
#include "list"
#include "vector"
#include <chrono>
using namespace std::chrono;

#define CAP_INCREMENT 4096

int_t timestep = 0, maxtimestep=60000;

int_t
    n_field = 0,        // Number of actual particles
    n_virt = 0,         // Number of virtual (ghost) particles
    n_capacity = 0,     // Size of paricle list (arbitrary)
    n_pairs = 0,        // Number of pairs
    n_pair_cap = 0;     // Size of pair list

particle_t *p_list = NULL;    // Global list of actual and virtual particles
pair_t *pairs = NULL;       // Global list of interacting pairs


std::time_t result = std::time(nullptr);

#define R (scale_k*H)
#define N_CELL_X (int_t)((B+2*R)/R)
#define N_CELL_Y 1
#define CID(X,Y) (X + Y * N_CELL_X)

typedef struct cell_t{
    std::vector<particle_t*> p_vector;
}cell_t;

int_t square_count = 0;
std::vector<cell_t> cell_list (300);

void
int_force ( void )
{
    //#pragma omp parallel for
    for ( int_t k=0; k<(n_field+n_virt); k++ )
        INDVXDT(k,0) = INDVXDT(k,1) = 0.0;

    //#pragma omp parallel for
    for ( int_t k=0; k<n_field; k++ )
        if ( INTER(k) < free_surface )
            RHO(k) = density;

    // Equations of state
    //#pragma omp parallel for
    for ( int_t k=0; k<(n_field+n_virt); k++ )
        P(k) = sos * sos * density * ((pow(RHO(k)/density,7.0)-1.0) / 7.0);

    // All the pairwise interactions
    //#pragma omp parallel for
    for ( int_t kk=0; kk<n_pairs; kk++ )
    {
        int_t
            i = pairs[kk].i,
            j = pairs[kk].j;
        real_t hx, hy;

        // i acts on j
        hx=-( P(i)/pow(RHO(i),2) + P(j)/pow(RHO(j),2) ) * pairs[kk].dwdx[0];
	    hy=-( P(i)/pow(RHO(i),2) + P(j)/pow(RHO(j),2) ) * pairs[kk].dwdx[1];
        //#pragma omp atomic
        INDVXDT(i,0) += M(j) * hx;
        //#pragma omp atomic
        INDVXDT(i,1) += M(j) * hy;

        // j acts on i, reverse sign because dwdx is X(i)-X(j)
        hx=-( P(j)/pow(RHO(j),2) + P(i)/pow(RHO(i),2) )*(-pairs[kk].dwdx[0]);
	    hy=-( P(j)/pow(RHO(j),2) + P(i)/pow(RHO(i),2) )*(-pairs[kk].dwdx[1]);
        //#pragma omp atomic
        INDVXDT(j,0) += M(i) * hx;
        //#pragma omp atomic
        INDVXDT(j,1) += M(i) * hy;
    }
}


void
kernel ( void )
{
    static real_t factor = 7.0 / (478.0 * M_PI * H * H);
    //#pragma omp parallel for
    for ( int_t kk=0; kk<n_pairs; kk++ )
    {
        pair_t *p = &pairs[kk];  // convenience alias
        real_t q = p->q;
        real_t dx[2] = {X(p->i)-X(p->j), Y(p->i)-Y(p->j)};
        //printf("pair is %ld, %ld\n", p->i, p->j);


        if ( q == 0.0 )
        {
            p->w = factor * (
                pow((3-q),5) - 6*pow((2-q),5) + 15*pow((1-q),5)
            );
            p->dwdx[0] = p->dwdx[1] = 0.0;
        }
        else if ( q>0.0 && q<=1.0 )
        {
            p->w = factor * (
                pow((3-q),5) - 6*pow((2-q),5) + 15*pow((1-q),5)
            );
            p->dwdx[0] = (factor/pow(H,2)) *
                (-120+120*pow(q,2)-50*pow(q,3))*dx[0];
            p->dwdx[1] = (factor/pow(H,2)) *
                (-120+120*pow(q,2)-50*pow(q,3))*dx[1];
        }
        else if ( q>1.0 && q<=2.0 )
        {
            p->w = factor * ( pow(3-q,5) - 6*pow(2-q,5));
            p->dwdx[0] = (factor/H) *
                ((-5)*pow((3-q),4)+30*pow((2-q),4))*(dx[0]/p->r);
            p->dwdx[1] = (factor/H) *
                ((-5)*pow((3-q),4)+30*pow((2-q),4))*(dx[1]/p->r);
        
        }
        else if ( q>2.0 && q<=3.0 )
        {
            p->w = factor * pow(3-q,5);
            p->dwdx[0] = (factor/H) * ((-5)*pow((3-q),4))*(dx[0]/p->r);
            p->dwdx[1] = (factor/H) * ((-5)*pow((3-q),4))*(dx[1]/p->r);
        
        }
        else
        {
            p->w = 0.0;
            p->dwdx[0] = p->dwdx[1] = 0.0;
        }
        //#pragma omp atomic
        WSUM(p->i) += p->w;
        //#pragma omp atomic
        WSUM(p->j) += p->w;
    }
}

void
set_particle_bucket (void){
    int_t n_total = n_field + n_virt;
    for (int_t i = 0; i < n_total; i++)
    {
        particle_t *particle = &p_list[i];
        int_t x_pos = (X(i) / B) * N_CELL_X;
        int_t y_pos = (Y(i) / INT64_MAX) * N_CELL_Y; 
        
        particle->id = i;
        particle->cell_x = x_pos;
        particle->cell_y = y_pos;
        
        cell_list[CID(x_pos,y_pos)].p_vector.push_back(particle);
        
        // printf("Particle %ld added to cell: %ld\n", i , CID(x_pos, y_pos));
        
    }
}
void
find_pairs (particle_t * particle, int cx, int cy, int id){
    if (cx >= N_CELL_X || cx < 0 ||
        cy >= N_CELL_Y || cy < 0 || particle == NULL) return;
    cell_t current_cell = cell_list[CID(cx,cy)];
    for (int  j = 0; j < current_cell.p_vector.size(); j++)
    {
        particle_t * particle_pair = current_cell.p_vector[j];
        if (particle->id >= particle_pair->id) continue;
        //printf("hello");


        real_t dist_sq = 
                        (particle->x[0]-particle_pair->x[0]) *
                        (particle->x[0]-particle_pair->x[0]) + 
                        (particle->x[1]-particle_pair->x[1]) *
                        (particle->x[1]-particle_pair->x[1]);
        // printf("dst: %lf, x: %lf %lf y: %lf %lf\n", dist_sq,particle->x[0],particle_pair->x[0],particle->x[1],particle_pair->x[1]);
        // printf("IDS: %ld, %ld \n", particle->id, particle_pair->id);
        if ( dist_sq < (R)*(R) )
        {
            //#pragma omp critical
            //{
            INTER(particle->id) += 1;
            INTER(particle_pair->id) += 1;
            //if (particle->cell_x != particle_pair->cell_x) printf("Particles are in %ld, %ld\n", particle->cell_x, particle_pair->cell_x);
            //printf("%ld\n", n_pairs);

            int_t kk = n_pairs;
            pairs[kk].i = particle->id;
            pairs[kk].j = particle_pair->id;
            pairs[kk].r = sqrt(dist_sq);
            pairs[kk].q = pairs[kk].r / H;
            pairs[kk].w = 0.0;
            pairs[kk].dwdx[0] = pairs[kk].dwdx[1] = 0.0;
            n_pairs += 1;
            if ( n_pairs == n_pair_cap )
                resize_pair_list();
            //}
        }
    }
}

void
new_find_neighbours (void){
    for (int  j = 0; j < cell_list.size(); j++) {
        cell_list[j].p_vector.clear();
    }
    set_particle_bucket();
    
    int_t n_total = n_field + n_virt;
    n_pairs = 0;
    
    for ( int_t k=0; k<n_total; k++ )

        INTER(k) = WSUM(k) = AVRHO(k) = 0;

    for ( int_t i=0; i<n_total; i++ )
    {
        particle_t *particle = &p_list[i];

        int cx = particle->cell_x;
        int cy = particle->cell_y;

        // find_pairs(particle,cx-1,cy-1,i);
        find_pairs(particle,cx-1,cy,i);
        // find_pairs(particle,cx-1,cy+1,i);

        // find_pairs(particle,cx,cy-1,i);
        find_pairs(particle,cx,cy,i);
        // find_pairs(particle,cx,cy+1,i);

        // find_pairs(particle,cx+1,cy-1,i);
        find_pairs(particle,cx+1,cy,i);
        // find_pairs(particle,cx+1,cy+1,i);
    
    }

    // for (int i = 0; i < cell_list.size(); i++)
    // {
    //     if(cell_list[i].p_vector.size() > 0)
    //     printf("Cell %ld contains %ld particles\n", i, cell_list[i].p_vector.size());
    // }
    // printf("Number of pairs: %ld\n", n_pairs);
    // printf("Number of particles: %ld\n", n_total);
    

}

void
find_neighbors ( void )
{
    int_t n_total = n_field + n_virt;
    n_pairs = 0;

    //#pragma omp parallel for
    for ( int_t k=0; k<n_total; k++ )
        INTER(k) = WSUM(k) = AVRHO(k) = 0;

    for ( int_t i=0; i<n_total-1; i++ )
    {
        //#pragma omp parallel for
        for ( int_t j=i+1; j<n_total; j++ )
        {
            real_t dist_sq = (X(i)-X(j))*(X(i)-X(j)) + (Y(i)-Y(j))*(Y(i)-Y(j));
            //printf("dst: %lf\n", dist_sq);
            if ( dist_sq < (scale_k*H)*(scale_k*H) )
            {
                //#pragma omp critical
                {
                    INTER(i) += 1;
                    INTER(j) += 1;
                    int_t kk = n_pairs;
                    pairs[kk].i = i;
                    pairs[kk].j = j;
                    pairs[kk].r = sqrt(dist_sq);
                    pairs[kk].q = pairs[kk].r / H;
                    pairs[kk].w = 0.0;
                    pairs[kk].dwdx[0] = pairs[kk].dwdx[1] = 0.0;
                    n_pairs += 1;
                    if ( n_pairs == n_pair_cap )
                        resize_pair_list();
                }
            }
        }
    }
    //printf("Number of pairs: %ld\n", n_pairs);
}


void
cont_density ( void )
{
    for ( int_t k=0; k<(n_field+n_virt); k++ )
        DRHODT(k) = 0.0;

    //#pragma omp parallel for
    for ( int_t kk=0; kk<n_pairs; kk++ )
    {
        int_t
            i = pairs[kk].i,
            j = pairs[kk].j;
        real_t vcc;

        vcc = (VX(i)-VX(j))*pairs[kk].dwdx[0] +
              (VY(i)-VY(j))*pairs[kk].dwdx[1];
        //#pragma omp atomic
        DRHODT(i) += RHO(i) * (M(j)/RHO(j)) * vcc;

        // Reverse sign of dwdx because it is calc. with X(i)-X(j)
        vcc = (VX(j)-VX(i))*(-pairs[kk].dwdx[0]) +
              (VY(j)-VY(i))*(-pairs[kk].dwdx[1]);
        //#pragma omp atomic
        DRHODT(j) += RHO(j) * (M(i)/RHO(i)) * vcc;
    }

    //#pragma omp parallel for
    for ( int_t k=0; k<(n_field+n_virt); k++ )
    {
        RHO(k) += 0.5 * dt * DRHODT(k);
    }
}


void
ext_force ( void )
{
    //#pragma omp parallel for
    for ( int_t k=0; k<n_field; k++ )
        EXDVXDT(k,1) = -9.81;
}


void
correction ( void )
{
    //#pragma omp parallel for
    for ( int_t kk=0; kk<n_pairs; kk++ )
    {
        int_t
            i = pairs[kk].i,
            j = pairs[kk].j;
        real_t drho;
        drho = RHO(i)-RHO(j);
        //#pragma omp atomic
        AVRHO(i) -= drho * pairs[kk].w / WSUM(i);
        drho = RHO(j)-RHO(i);
        //#pragma omp atomic
        AVRHO(j) -= drho * pairs[kk].w / WSUM(j);
    }

    //#pragma omp parallel for
    for ( int_t k=0; k<(n_field+n_virt); k++ )
    {
        if ( TYPE(k) < 0 && INTER(k) < 10 )
            RHO(k) = density;
        else
            RHO(k) += 0.5 * AVRHO(k);
    }
}

//----------------------My Code --------------------



void
time_step ( void )
{
    auto start = high_resolution_clock::now();
    new_find_neighbours();
    // Together, these implement direct_find from proto
    // find_neighbors();
    auto stop  = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);
 
    // To get the value of duration use the count()
    // member function on the duration object
    //printf("time: %ld\n", duration.count());
    kernel();

    if ( (timestep%20)==0 )
    {
        static double start, end;
        if ( timestep == 0 )
            start= result;
        end = result;
        printf ( "Timestep %ld\ttime %.2e s\t\t", timestep, end-start );
        printf ( "%ld actual, %ld virtuals, %ld pairs\n", n_field, n_virt, n_pairs );
        start = result;
    }
    // Density
    cont_density();

    if ( timestep > 0 )
        correction();

    // Internal and external forces
    int_force();
    ext_force();

    //#pragma omp parallel for
    for ( int_t k=0; k<n_field; k++ )
    {
        DVX(k,0) = INDVXDT(k,0) + EXDVXDT(k,0); //+ ardvxdt
        DVX(k,1) = INDVXDT(k,1) + EXDVXDT(k,1); //+ ardvxdt
    }

}


void
time_integration ( void )
{

    for ( timestep=0; timestep<maxtimestep; timestep++ )
    {
        if ( timestep > 0 )
        {
            //#pragma omp parallel for
            for ( int_t k=0; k<n_field; k++ )
            {
                // Vx, Vy cloned to vx_min vy_min for some reason
                // Omitting this until I see the purpose
                VX(k) += 0.5 * dt * DVX(k,0);
                VY(k) += 0.5 * dt * DVX(k,1);
                X(k) += dt * VX(k);
                Y(k) += dt * VY(k);
            }
        }

        generate_virtual_particles();
        time_step();

        if ( timestep == 0 )
        {
            //#pragma omp parallel for
            for ( int_t k=0; k<n_field; k++ )
            {
                // Calculate initial distributions
                RHO(k) += 0.5 * dt * DRHODT(k);
                VX(k) += 0.5 * dt * DVX(k,0);
                VY(k) += 0.5 * dt * DVX(k,1);
                X(k) += dt * VX(k);
                Y(k) += dt * VY(k);
            }
        }
        else
        {
            //#pragma omp parallel for
            for ( int_t k=0; k<(n_field+n_virt); k++ )
            {
                RHO(k) += 0.5 * dt * DRHODT(k);
                if ( k < n_field )
                {
                    VX(k) += 0.5 * dt * DVX(k,0);
                    VY(k) += 0.5 * dt * DVX(k,1);
                    X(k) += dt * VX(k);
                    Y(k) += dt * VY(k);

                    // Reflect velocity at boundaries
                    if ( Y(k) < 0.0 )
                        VY(k) = -VY(k);
                    if ( X(k) > B )
                        VX(k) = -VX(k);
                    if ( X(k) < 0.0 )
                        VX(k) = -VX(k);
                }
            }
        }

        if ( (timestep % 200) == 0 )
        {
            static int_t printcount = 0;
            char fname[256];
            sprintf ( fname, "data/%.5ld.txt", printcount );
            FILE *out;
            out = fopen ( fname, "w" );
            write_positions ( out, 0, n_field );
            fclose ( out );
            printcount += 1;
        }
    }
}


int
main ( int argc, char **argv )
{
    initialize();
    time_integration();
    finalize();
    exit ( EXIT_SUCCESS );
}


// Identify particles near tank boundary, set up virtual counterparts
// Particles are near boundary when they are within 1.55 * H of tank
void
generate_virtual_particles ( void )
{
    // Article sec. 2.3: ghost particles are mirror images wrt. wall
    // Vertical distance less than 1.55H (no euclid here)
    // Field variables are as per Dirichlet (interp.) or Neumann (reflect)
    // Neumann implemented here for the time being

    n_virt = 0; // No virtual particles to begin with

    real_t boundary = 1.55 * H;
    for ( int_t k=0; k<n_field; k++ )
    {
        // 5 cases will never add more than 5 particles
        // Increase list length if we are within 6 of capacity
        if ( (n_field + n_virt) >= (n_capacity-6) )
            resize_particle_list();

        // Horizontal mirror left
        if ( X(k) < boundary )
        {
            int_t gk = n_field + n_virt;
            n_virt += 1;
            X(gk) = -X(k), VX(gk) = -VX(k);
            Y(gk) = Y(k),  VY(gk) = VY(k);
            P(gk) = P(k), RHO(gk) = RHO(k), M(gk) = M(k); // Neumann boundary
            TYPE(gk) = -2, HSML(gk) = H;
        }

        // Horizontal mirror right
        if ( X(k) > B-boundary )
        {
            int_t gk = n_field + n_virt;
            n_virt += 1;
            X(gk) = 2*B-X(k), VX(gk) = -VX(k);
            Y(gk) = Y(k), VY(gk) = VY(k);
            P(gk) = P(k), RHO(gk) = RHO(k), M(gk) = M(k); // Neumann boundary
            TYPE(gk) = -2, HSML(gk) = H;
        }

        // Vertical mirror bottom
        if ( Y(k) < boundary )
        {
            int_t gk = n_field + n_virt;
            n_virt += 1;
            X(gk) = X(k),  VX(gk) = VX(k);
            Y(gk) = -Y(k), VY(gk) = -VY(k);
            P(gk) = P(k), RHO(gk) = RHO(k), M(gk) = M(k); // Neumann boundary
            TYPE(gk) = -2, HSML(gk) = H;
        }

        // Lower left corner
        if ( X(k) < boundary && Y(k) < boundary )
        {
            int_t gk = n_field + n_virt;
            n_virt += 1;
            X(gk) = -X(k), VX(gk) = -VX(k);
            Y(gk) = -Y(k), VY(gk) = -VY(k);
            P(gk) = P(k), RHO(gk) = RHO(k), M(gk) = M(k); // Neumann boundary
            TYPE(gk) = -2, HSML(gk) = H;
        }

        // Lower right corner
        if ( X(k) > B-boundary && Y(k) < boundary )
        {
            int_t gk = n_field + n_virt;
            n_virt += 1;
            X(gk) = 2*B-X(k), VX(gk) = -VX(k);
            Y(gk) = -Y(k), VY(gk) = -VY(k);
            P(gk) = P(k), RHO(gk) = RHO(k), M(gk) = M(k); // Neumann boundary
            TYPE(gk) = -2, HSML(gk) = H;
        }
    }
}


void
initialize ( void )
{
    //-----------------------My code---------------------------
    square_count = (N_CELL_X) * N_CELL_Y;
    printf("Squares: %ld \n", square_count);
    //cell_list = (std::list<int>*) malloc ( square_count * sizeof(std::list<int>*) );

    
    
    

    // Calculate number of particles in problem according to parameters
    // (This corresponds to 'total_particle_amount' in F90 benchmark)

    // Number of particles in the initial, rectangular dam shape:
    int_t n[2] = { (int_t)(1 + L/DELTA), (int_t)(1 + T/DELTA) };

// Testing value: full basin at rest
//    int_t n[2] = { 1 + B/DELTA, 1 + T/DELTA };

    n_field = n[0] * n[1];  // This value remains constant (actual particles)

    // Allocate empty list to closest higher list capacity multiple
    n_capacity = CAP_INCREMENT *
        ceil( (n_field+n_virt) / (real_t)CAP_INCREMENT );
    p_list = (particle_t *) malloc ( n_capacity * sizeof(particle_t) );
    memset ( p_list, 0, n_capacity * sizeof(particle_t) );

//fprintf ( stderr, "Initial allocation %ld\n", n_capacity );

    // Set up the initial list of particles
    // (This corresponds to 'ini_particle_distribution' in F90 benchmark)
    
    // Coordinates of the initial distribution:
    for ( int_t j=0; j<n[0]; j++ )      // j is column index
    {
        for ( int_t i=0; i<n[1]; i++ )  // i is row index
        {
            int_t k = j * n[1] + i; // k is particle index
            X(k) = H + j * DELTA;   // Initial 2D position of dam, offset
            Y(k) = H + i * DELTA;   // by H not to collide with boundary
        }
    }

    // Properties of the initial distribution:
    for ( int_t k=0; k<n_field; k++ )
    {
        VX(k) = VY(k) = 0.0;    // No initial velocity
        RHO(k) = density;       // Parameter value

        // Static pressure combines gravitation (9.81) and vertical position
        P(k) = density * 9.81 * (T - Y(k) );

        // Distribute total mass on number of particles
        M(k) = L * T * density / (real_t) n_field;

        TYPE(k) = 2;   // Actual particle
        HSML(k) = H;   // Resolution
    }

    // Set up space for list of pairs (arbitrary, resized on need)
    n_pair_cap = 64 * (n_field+n_virt);
    pairs = (pair_t*)malloc ( n_pair_cap * sizeof ( pair_t ) );
}


void
finalize ( void )
{
    free ( p_list );
    free ( pairs );
}


void
resize_pair_list ( void )
{
    n_pair_cap += CAP_INCREMENT;
    pairs = (pair_t*)realloc ( pairs, n_pair_cap * sizeof(pair_t) );
    fprintf ( stderr, "Resized to %ld pairs\n", n_pair_cap );
}


void
resize_particle_list ( void )
{
    n_capacity += CAP_INCREMENT;
    p_list = (particle_t*)realloc ( p_list, n_capacity * sizeof(particle_t) );
    fprintf ( stderr, "Resized to %ld particles\n", n_capacity );
}


void
write_positions ( FILE *out, int_t start, int_t end )
{
    fprintf ( out,
        "variables=\"k\",\"x\",\"y\",\"vx\",\"vy\",\"pressure\"\n"
        "zone i=%ld j=%d\n", end-start, 1
    );
    for ( int_t k=start; k<end; k++ )
        fprintf ( out, INT_FMT "\t"
            REAL_FMT "\t" REAL_FMT "\t" REAL_FMT "\t" REAL_FMT "\t" REAL_FMT "\n" ,
            k, X(k), Y(k), VX(k), VY(k), P(k)
        );
}


void
write_state ( FILE *out, int_t start, int_t end )
{
    for ( int_t k=start; k<end; k++ )
        fprintf ( out, INT_FMT "\t" REAL_FMT "\t" REAL_FMT "\t" REAL_FMT "\n",
            k, M(k), RHO(k), P(k)
        );
}
