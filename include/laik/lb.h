/**
 * Types and function signatures for the load balancing extension.
 * 
 * No copyright notice for now I suppose.
 */
#include "core.h"
#include "space.h"

// enum defining current load balancing state (before / after iteration)
// start: record starting time and return
// stop : calculate time difference (for weights) and create new partitioning
typedef enum {
    START_LB_SEGMENT,
    STOP_LB_SEGMENT
} Laik_LBState;

// enum defining currently available load balancing algorithms
typedef enum {
    LB_RCB,
    LB_MORTON,
    LB_HILBERT
} Laik_LBAlgorithm;

//////////////////////
// sfc partitioners //
//////////////////////

// main morton (z-curve) function
void runMortonPartitioner(Laik_RangeReceiver *r, Laik_PartitionerParams *p);

// create new morton partitioner
Laik_Partitioner *laik_new_morton_partitioner(double *tweights);

/////////////////////
// rcb partitioner //
/////////////////////

// main rcb function
void runRCBPartitioner(Laik_RangeReceiver *r, Laik_PartitionerParams *p);

// create new rcb partitioner
Laik_Partitioner *laik_new_rcb_partitioner(double *weights);

////////////////////
// load balancing //
////////////////////

// collect weights into a designated array
//
// the size of the array is the size of the laik space
// each index in this array corresponds to the element's weight
//
// this uses the aggregation functionality shown in the 1d vector sum example
//
// returns null if it's the first iteration or a pointer to the weight array otherwise
double* laik_lb_measure(Laik_Partitioning *group, double ttime);

// create and return new partitioning based on time measurements using an input-chosen load balancing algorithm
//
// TODO: disable balancing under a certain threshold
Laik_Partitioning *laik_lb_balance(Laik_LBState state, Laik_Partitioning *partitioning, Laik_LBAlgorithm algorithm /*, double threshold*/);