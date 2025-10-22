/* This file is part of the LAIK parallel container library.
 * Copyright (c) 2025 Flavius Schmidt
 *
 * LAIK is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 3.
 *
 * LAIK is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */


/**
 * Debug logging functions for outputting partitioning borders and calling external Python scripts.
 * This assumes that the code is being called from its default location in the project repository.
 */

 
#ifndef LAIK_LB_VIS_H
#define LAIK_LB_VIS_H

#include "core.h"

// export ranges associated with tasks to csv file inside lbviz directory
// expects the filename as input, not the full path
//
// format (header):
// - 1d: from_x, to_x,                             task
// - 2d: from_x, to_x, from_y, to_y,               task
// - 3d: from_x, to_x, from_y, to_y, from_z, to_z, task
void laik_lbvis_export_partitioning(const char *filename, Laik_RangeList *lr);

// call task visualization script from inside example directory automatically
void laik_lbvis_visualize_partitioning(const char *filename);

// purge json data and images
void laik_lbvis_remove_visdata();

// plot program trace through external script
void laik_lbvis_save_trace();

// enable svg trace
void laik_lbvis_enable_trace(int id, Laik_Instance *inst);

#endif