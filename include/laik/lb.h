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
    LB_HILBERT
} Laik_LBAlgorithm;

// get algorithm string from enum
const char *laik_get_lb_algorithm_name(Laik_LBAlgorithm algo);

//////////////////////
// sfc partitioners //
//////////////////////

// main hilbert curve function
void runHilbertPartitioner(Laik_RangeReceiver *r, Laik_PartitionerParams *p);

// create new hilbert curve partitioner
Laik_Partitioner *laik_new_hilbert_partitioner(double *tweights);

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

// create and return new partitioning based on time measurements using an input-chosen load balancing algorithm
//
// when under a specific stopping threshold for a certain number of consecutive runs, load balancing stops
// when over a specific starting thershold for a certain number of consecutive runs, load balancing restarts
// two thresholds were chosen alongside patience counters to avoid oscillation
//
// returns null for the starting call, the same partitioning if nothing was changed (do not free this if you intend on using it afterwards!)
// or a different partitioning if something changed
Laik_Partitioning *laik_lb_balance(Laik_LBState state, Laik_Partitioning *partitioning, Laik_LBAlgorithm algorithm /*, double threshold*/);