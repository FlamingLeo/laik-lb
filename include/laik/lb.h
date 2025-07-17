/**
 * Types and function signatures for the load balancing extension.
 * 
 * TODO: Make comments better.
 */
#include "core.h"
#include "space.h"

// simple range (*space, from / to indices) associated with weight
typedef struct RangeWeight
{
    Laik_Range range;
    double weight;
} RangeWeight;

// data used to create and run an RCB partitioner
typedef struct RCBData
{
    RangeWeight *rweights; // array of rangeweights
    size_t count;          // rweights length
    unsigned tidcount;
    int dims;
} RCBData;

// main rcb function
void runRCBPartitioner(Laik_RangeReceiver *r, Laik_PartitionerParams *p);

// create new rcb partitioner
Laik_Partitioner *laik_new_rcb_partitioner(RangeWeight *rweights, unsigned count, unsigned tidcount, int dims);

// collect arrival times of each task into a designated array
//
// the size of the array is the size of the laik group
// each index in this array corresponds to the task's index
//
// this uses the aggregation functionality shown in the 1d vector sum example
//
// returns 1 if it's the first time measuring and 0 otherwise
int laik_lb_measure(Laik_Group *group, double *timearr);

// create and return new partitioning based on time measurements (current: RCB)
//
// TODO: disable balancing under a certain threshold
Laik_Partitioning *laik_lb_balance(Laik_Partitioning *partitioning /*, Laik_Partitioner *algorithm, double threshold*/);