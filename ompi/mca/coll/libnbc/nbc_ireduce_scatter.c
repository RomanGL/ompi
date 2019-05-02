/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2006      The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2006      The Technical University of Chemnitz. All
 *                         rights reserved.
 * Copyright (c) 2013-2015 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2014-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2015      The University of Tennessee and The University
 *                         of Tennessee Research Foundation. All rights
 *                         reserved.
 * Copyright (c) 2017      IBM Corporation.  All rights reserved.
 * Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *
 */
#include "opal/align.h"
#include "opal/util/bit_ops.h"
#include "ompi/mca/coll/base/coll_base_util.h"

#include "nbc_internal.h"

static inline int reduce_scatter_pairwise_exchange(
  const void *sendbuf, void *recvbuf, const int *recvcounts, MPI_Datatype datatype,
  MPI_Op op, struct ompi_communicator_t *comm, ompi_coll_libnbc_module_t *libnbc_module,
  ompi_request_t ** request, bool persistent);

static inline int reduce_scatter_butterfly(
  const void *sendbuf, void *recvbuf, const int *recvcounts, MPI_Datatype datatype,
  MPI_Op op, struct ompi_communicator_t *comm, ompi_coll_libnbc_module_t *module,
  ompi_request_t ** request, bool persistent);

static int ompi_sum_counts(const int *counts, int *displs, int nprocs_rem, int lo, int hi);

/* an reduce_csttare schedule can not be cached easily because the contents
 * ot the recvcounts array may change, so a comparison of the address
 * would not be sufficient ... we simply do not cache it */

static int nbc_reduce_scatter_init(const void* sendbuf, void* recvbuf, const int *recvcounts, MPI_Datatype datatype,
                                   MPI_Op op, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                   struct mca_coll_base_module_2_3_0_t *module, bool persistent) {
  int comm_size, res, totalcount;
  char inplace;  
  ompi_coll_libnbc_module_t *libnbc_module = (ompi_coll_libnbc_module_t*) module;

  enum { NBC_REDUCE_SCAT_PAIRWISE_EXCHANGE, NBC_REDUCE_SCAT_BUTTERFLY } alg;

  NBC_IN_PLACE(sendbuf, recvbuf, inplace);
  comm_size = ompi_comm_size (comm);

  if (libnbc_ireduce_scatter_algorithm == 0) {
    alg = NBC_REDUCE_SCAT_PAIRWISE_EXCHANGE;
  } else {
    /* user forced dynamic decision */
    if (libnbc_ireduce_scatter_algorithm == 1) {
      alg = NBC_REDUCE_SCAT_PAIRWISE_EXCHANGE;
    } else if (libnbc_ireduce_scatter_algorithm == 2) {
      alg = NBC_REDUCE_SCAT_BUTTERFLY;
    } else {
      alg = NBC_REDUCE_SCAT_PAIRWISE_EXCHANGE;
    }
  }

  totalcount = 0;
  for (int r = 0 ; r < comm_size ; ++r) {
    totalcount += recvcounts[r];
  }

  if ((1 == comm_size && (!persistent || inplace)) || 0 == totalcount) {
    if (!inplace) {
      /* single node not in_place: copy data to recvbuf */
      res = NBC_Copy(sendbuf, recvcounts[0], datatype, recvbuf, recvcounts[0], datatype, comm);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }

    return nbc_get_noop_request(persistent, request);
  }

  switch (alg) {
    case NBC_REDUCE_SCAT_PAIRWISE_EXCHANGE:
      res = reduce_scatter_pairwise_exchange(sendbuf, recvbuf, recvcounts, datatype, 
                                             op, comm, libnbc_module, request, persistent);
      break;
    case NBC_REDUCE_SCAT_BUTTERFLY:
      res = reduce_scatter_butterfly(sendbuf, recvbuf, recvcounts, datatype, 
                                     op, comm, libnbc_module, request, persistent);
      break;
  }

  return res;
}

/*
 * ompi_coll_base_reduce_scatter_intra_butterfly
 *
 * Function:  Butterfly algorithm for reduce_scatter
 * Accepts:   Same as MPI_Reduce_scatter
 * Returns:   MPI_SUCCESS or error code
 *
 * Description:  Implements butterfly algorithm for MPI_Reduce_scatter [*].
 *               The algorithm can be used both by commutative and non-commutative
 *               operations, for power-of-two and non-power-of-two number of processes.
 *
 * [*] J.L. Traff. An improved Algorithm for (non-commutative) Reduce-scatter
 *     with an Application // Proc. of EuroPVM/MPI, 2005. -- pp. 129-137.
 *
 * Time complexity: O(m\lambda + log(p)\alpha + m\beta + m\gamma),
 *   where m = sum of rcounts[], p = comm_size
 * Memory requirements (per process): 2 * m * typesize + comm_size
 *
 * Example: comm_size=6, nprocs_pof2=4, nprocs_rem=2, rcounts[]=1, sbuf=[0,1,...,5]
 * Step 1. Reduce the number of processes to 4
 * rank 0: [0|1|2|3|4|5]: send to 1: vrank -1
 * rank 1: [0|1|2|3|4|5]: recv from 0, op: vrank 0: [0|2|4|6|8|10]
 * rank 2: [0|1|2|3|4|5]: send to 3: vrank -1
 * rank 3: [0|1|2|3|4|5]: recv from 2, op: vrank 1: [0|2|4|6|8|10]
 * rank 4: [0|1|2|3|4|5]: vrank 2: [0|1|2|3|4|5]
 * rank 5: [0|1|2|3|4|5]: vrank 3: [0|1|2|3|4|5]
 *
 * Step 2. Butterfly. Buffer of 6 elements is divided into 4 blocks.
 * Round 1 (mask=1, nblocks=2)
 * 0: vrank -1
 * 1: vrank  0 [0 2|4 6|8|10]: exch with 1: send [2,3], recv [0,1]: [0 4|8 12|*|*]
 * 2: vrank -1
 * 3: vrank  1 [0 2|4 6|8|10]: exch with 0: send [0,1], recv [2,3]: [**|**|16|20]
 * 4: vrank  2 [0 1|2 3|4|5] : exch with 3: send [2,3], recv [0,1]: [0 2|4 6|*|*]
 * 5: vrank  3 [0 1|2 3|4|5] : exch with 2: send [0,1], recv [2,3]: [**|**|8|10]
 *
 * Round 2 (mask=2, nblocks=1)
 * 0: vrank -1
 * 1: vrank  0 [0 4|8 12|*|*]: exch with 2: send [1], recv [0]: [0 6|**|*|*]
 * 2: vrank -1
 * 3: vrank  1 [**|**|16|20] : exch with 3: send [3], recv [2]: [**|**|24|*]
 * 4: vrank  2 [0 2|4 6|*|*] : exch with 0: send [0], recv [1]: [**|12 18|*|*]
 * 5: vrank  3 [**|**|8|10]  : exch with 1: send [2], recv [3]: [**|**|*|30]
 *
 * Step 3. Exchange with remote process according to a mirror permutation:
 *         mperm(0)=0, mperm(1)=2, mperm(2)=1, mperm(3)=3
 * 0: vrank -1: recv "0" from process 0
 * 1: vrank  0 [0 6|**|*|*]: send "0" to 0, copy "6" to rbuf (mperm(0)=0)
 * 2: vrank -1: recv result "12" from process 4
 * 3: vrank  1 [**|**|24|*]
 * 4: vrank  2 [**|12 18|*|*]: send "12" to 2, send "18" to 3, recv "24" from 3
 * 5: vrank  3 [**|**|*|30]: copy "30" to rbuf (mperm(3)=3)
 */
static inline int reduce_scatter_butterfly(
  const void *sendbuf, void *recvbuf, const int *recvcounts, MPI_Datatype datatype,
  MPI_Op op, struct ompi_communicator_t *comm, ompi_coll_libnbc_module_t *module,
  ompi_request_t ** request, bool persistent)
{
  NBC_Schedule *schedule;
  char *tmpbuf = NULL, *psend, *precv;
  int *displs = NULL, index;
  ptrdiff_t span, gap, totalcount, extent;
  int err = MPI_SUCCESS;
  int comm_size = ompi_comm_size(comm);
  int rank = ompi_comm_rank(comm);

  if (comm_size < 2)
      return MPI_SUCCESS;

  schedule = OBJ_NEW(NBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    err = OMPI_ERR_OUT_OF_RESOURCE;
    goto cleanup_and_return;
  }

  displs = malloc(sizeof(*displs) * comm_size);
  if (NULL == displs) {
      err = OMPI_ERR_OUT_OF_RESOURCE;
      goto cleanup_and_return;
  }
  displs[0] = 0;
  for (int i = 1; i < comm_size; i++) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
  }
  totalcount = displs[comm_size - 1] + recvcounts[comm_size - 1];

  ompi_datatype_type_extent(datatype, &extent);
  span = opal_datatype_span(&datatype->super, totalcount, &gap);
  tmpbuf = malloc(span * 2);
  if (NULL == tmpbuf) {
      err = OMPI_ERR_OUT_OF_RESOURCE;
      goto cleanup_and_return;
  }

  psend = tmpbuf - gap;
  precv = (tmpbuf + span) - gap;

  if (sendbuf != MPI_IN_PLACE) {
    err = NBC_Sched_copy(sendbuf, false, totalcount, datatype, psend, false, totalcount, datatype, schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
      goto cleanup_and_return;
    }
  } else {
    err = NBC_Sched_copy(recvbuf, false, totalcount, datatype, psend, false, totalcount, datatype, schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
      goto cleanup_and_return;
    }
  }

  /*
    * Step 1. Reduce the number of processes to the nearest lower power of two
    * p' = 2^{\floor{\log_2 p}} by removing r = p - p' processes.
    * In the first 2r processes (ranks 0 to 2r - 1), all the even ranks send
    * the input vector to their neighbor (rank + 1) and all the odd ranks recv
    * the input vector and perform local reduction.
    * The odd ranks (0 to 2r - 1) contain the reduction with the input
    * vector on their neighbors (the even ranks). The first r odd
    * processes and the p - 2r last processes are renumbered from
    * 0 to 2^{\floor{\log_2 p}} - 1. Even ranks do not participate in the
    * rest of the algorithm.
    */

  /* Find nearest power-of-two less than or equal to comm_size */
  int nprocs_pof2 = opal_next_poweroftwo(comm_size);
  nprocs_pof2 >>= 1;
  int nprocs_rem = comm_size - nprocs_pof2;
  int log2_size = opal_cube_dim(nprocs_pof2);

  int vrank = -1;
  if (rank < 2 * nprocs_rem) {
      if ((rank % 2) == 0) {
          /* Even process */
          err = NBC_Sched_send(psend, false, totalcount, datatype, rank + 1, schedule, false);
          if (OMPI_SUCCESS != err) { goto cleanup_and_return; }

          /* This process does not participate in the rest of the algorithm */
          vrank = -1;
      } else {
          /* Odd process */
          err = NBC_Sched_recv(precv, false, totalcount, datatype, rank - 1, schedule, true);
          if (OMPI_SUCCESS != err) { goto cleanup_and_return; }

          err = NBC_Sched_op(precv, false, psend, false, totalcount, datatype, op, schedule, true);
          if (OMPI_SUCCESS != err) { goto cleanup_and_return; }

          /* Adjust rank to be the bottom "remain" ranks */
          vrank = rank / 2;
      }
  } else {
      /* Adjust rank to show that the bottom "even remain" ranks dropped out */
      vrank = rank - nprocs_rem;
  }

  if (vrank != -1) {
      /*
        * Now, psend vector of size totalcount is divided into nprocs_pof2 blocks:
        * block 0:   recvcounts[0] and recvcounts[1] -- for process 0 and 1
        * block 1:   recvcounts[2] and recvcounts[3] -- for process 2 and 3
        * ...
        * block r-1: recvcounts[2*(r-1)] and recvcounts[2*(r-1)+1]
        * block r:   recvcounts[r+r]
        * block r+1: recvcounts[r+r+1]
        * ...
        * block nprocs_pof2 - 1: recvcounts[r+nprocs_pof2-1]
        */
      int nblocks = nprocs_pof2, send_index = 0, recv_index = 0;
      for (int mask = 1; mask < nprocs_pof2; mask <<= 1) {
        int vpeer = vrank ^ mask;
        int peer = (vpeer < nprocs_rem) ? vpeer * 2 + 1 : vpeer + nprocs_rem;

        nblocks /= 2;
        if ((vrank & mask) == 0) {
          /* Send the upper half of reduction buffer, recv the lower half */
          send_index += nblocks;
        } else {
          /* Send the upper half of reduction buffer, recv the lower half */
          recv_index += nblocks;
        }

        /* Send blocks: [send_index, send_index + nblocks - 1] */
        int send_count = ompi_sum_counts(recvcounts, displs, nprocs_rem,
                                          send_index, send_index + nblocks - 1);
        index = (send_index < nprocs_rem) ? 2 * send_index : nprocs_rem + send_index;
        ptrdiff_t sdispl = displs[index];

        /* Recv blocks: [recv_index, recv_index + nblocks - 1] */
        int recv_count = ompi_sum_counts(recvcounts, displs, nprocs_rem,
                                         recv_index, recv_index + nblocks - 1);
        index = (recv_index < nprocs_rem) ? 2 * recv_index : nprocs_rem + recv_index;
        ptrdiff_t rdispl = displs[index];

        err = NBC_Sched_send(psend + (ptrdiff_t)sdispl * extent, false, send_count,
                             datatype, peer, schedule, false);
        if (MPI_SUCCESS != err) { goto cleanup_and_return; }

        err = NBC_Sched_recv(precv + (ptrdiff_t)rdispl * extent, false, recv_count,
                             datatype, peer, schedule, true);
        if (MPI_SUCCESS != err) { goto cleanup_and_return; }

        if (vrank < vpeer) {
          /* precv = psend <op> precv */
          err = NBC_Sched_op(psend + (ptrdiff_t)rdispl * extent, false,
                             precv + (ptrdiff_t)rdispl * extent, false,
                             recv_count, datatype, op, schedule, true);

          if (MPI_SUCCESS != err) { goto cleanup_and_return; }

          char *p = psend;
          psend = precv;
          precv = p;
        } else {
          /* psend = precv <op> psend */
          err = NBC_Sched_op(precv + (ptrdiff_t)rdispl * extent, false,
                             psend + (ptrdiff_t)rdispl * extent, false,
                             recv_count, datatype, op, schedule, true);

          if (MPI_SUCCESS != err) { goto cleanup_and_return; }
        }
        send_index = recv_index;
      }

      /*
        * psend points to the result block [send_index]
        * Exchange results with remote process according to a mirror permutation.
        */
      int vpeer = ompi_mirror_perm(vrank, log2_size);
      int peer = (vpeer < nprocs_rem) ? vpeer * 2 + 1 : vpeer + nprocs_rem;
      index = (send_index < nprocs_rem) ? 2 * send_index : nprocs_rem + send_index;

      if (vpeer < nprocs_rem) {
          /*
            * Process has two blocks: for excluded process and own.
            * Send the first block to excluded process.
            */
          err = NBC_Sched_send(psend + (ptrdiff_t)displs[index] * extent, false,
                               recvcounts[index], datatype, peer - 1,
                               schedule, false);
          if (MPI_SUCCESS != err) { goto cleanup_and_return; }
      }

      /* If process has two blocks, then send the second block (own block) */
      if (vpeer < nprocs_rem)
          index++;
      if (vpeer != vrank) {
        err = NBC_Sched_send(psend + (ptrdiff_t)displs[index] * extent, false, recvcounts[index],
                             datatype, peer, schedule, false);
        if (MPI_SUCCESS != err) { goto cleanup_and_return; }

        err = NBC_Sched_recv(recvbuf, false, recvcounts[rank],
                             datatype, peer, schedule, true);
        if (MPI_SUCCESS != err) { goto cleanup_and_return; }
      } else {
        err = NBC_Sched_copy(psend + (ptrdiff_t)displs[rank] * extent, false,
                             recvcounts[rank], datatype,
                             recvbuf, false,
                             recvcounts[rank], datatype,
                             schedule, true);

        if (MPI_SUCCESS != err) { goto cleanup_and_return; }
      }

  } else {
      /* Excluded process: receive result */
      int vpeer = ompi_mirror_perm((rank + 1) / 2, log2_size);
      int peer = (vpeer < nprocs_rem) ? vpeer * 2 + 1 : vpeer + nprocs_rem;

      err = NBC_Sched_recv(recvbuf, false, recvcounts[rank],
                           datatype, peer, schedule, true);
                           
      if (OMPI_SUCCESS != err) { goto cleanup_and_return; }
  }

  err = NBC_Sched_commit (schedule);
  if (OMPI_SUCCESS != err) { goto cleanup_and_return; }

  err = NBC_Schedule_request(schedule, comm, module, persistent, request, tmpbuf);
  if (OMPI_SUCCESS != err) { goto cleanup_and_return; }

cleanup_and_return:
  if (displs)
    free(displs);

  if (OMPI_SUCCESS != err) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
  }

  return err;
}

/* binomial reduce to rank 0 followed by a linear scatter ...
 *
 * Algorithm:
 * pairwise exchange
 * round r:
 *  grp = rank % 2^r
 *  if grp == 0: receive from rank + 2^(r-1) if it exists and reduce value
 *  if grp == 1: send to rank - 2^(r-1) and exit function
 *
 * do this for R=log_2(p) rounds
 *
 */
static inline int reduce_scatter_pairwise_exchange(
  const void *sendbuf, void *recvbuf, const int *recvcounts, MPI_Datatype datatype,
  MPI_Op op, struct ompi_communicator_t *comm, ompi_coll_libnbc_module_t *libnbc_module,
  ompi_request_t ** request, bool persistent) 
{
  MPI_Aint ext;
  NBC_Schedule *schedule;
  int res, maxr, peer, totalcount;
  ptrdiff_t gap, span, span_align;
  void *tmpbuf;
  char *rbuf, *lbuf, *buf, *sbuf;
  int comm_size = ompi_comm_size(comm);
  int rank = ompi_comm_rank(comm);

  totalcount = 0;
  for (int r = 0 ; r < comm_size ; ++r) {
    totalcount += recvcounts[r];
  }

  res = ompi_datatype_type_extent (datatype, &ext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  res = OMPI_SUCCESS;
  maxr = (int) ceil ((log((double) comm_size) / LOG2));

  span = opal_datatype_span(&datatype->super, totalcount, &gap);
  span_align = OPAL_ALIGN(span, datatype->super.align, ptrdiff_t);
  tmpbuf = malloc (span_align + span);
  if (OPAL_UNLIKELY(NULL == tmpbuf)) {
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  rbuf = (char *)(-gap);
  lbuf = (char *)(span_align - gap);

  schedule = OBJ_NEW(NBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    free(tmpbuf);
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  for (int r = 1, firstred = 1 ; r <= maxr ; ++r) {
    if ((rank % (1 << r)) == 0) {
      /* we have to receive this round */
      peer = rank + (1 << (r - 1));
      if (peer < comm_size) {
        /* we have to wait until we have the data */
        res = NBC_Sched_recv(rbuf, true, totalcount, datatype, peer, schedule, true);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          OBJ_RELEASE(schedule);
          free(tmpbuf);
          return res;
        }

        /* this cannot be done until tmpbuf is unused :-( so barrier after the op */
        if (firstred) {
          /* take reduce data from the sendbuf in the first round -> save copy */
          res = NBC_Sched_op (sendbuf, false, rbuf, true, totalcount, datatype, op, schedule, true);
          firstred = 0;
        } else {
          /* perform the reduce in my local buffer */
          res = NBC_Sched_op (lbuf, true, rbuf, true, totalcount, datatype, op, schedule, true);
        }

        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          OBJ_RELEASE(schedule);
          free(tmpbuf);
          return res;
        }
        /* swap left and right buffers */
        buf = rbuf; rbuf = lbuf ; lbuf = buf;
      }
    } else {
      /* we have to send this round */
      peer = rank - (1 << (r - 1));
      if (firstred) {
        /* we have to send the senbuf */
        res = NBC_Sched_send (sendbuf, false, totalcount, datatype, peer, schedule, false);
      } else {
        /* we send an already reduced value from lbuf */
        res = NBC_Sched_send (lbuf, true, totalcount, datatype, peer, schedule, false);
      }
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        free(tmpbuf);
        return res;
      }

      /* leave the game */
      break;
    }
  }

  res = NBC_Sched_barrier(schedule);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

  /* rank 0 is root and sends - all others receive */
  if (rank == 0) {
    for (long int r = 1, offset = 0 ; r < comm_size ; ++r) {
      offset += recvcounts[r-1];
      sbuf = lbuf + (offset*ext);
      /* root sends the right buffer to the right receiver */
      res = NBC_Sched_send (sbuf, true, recvcounts[r], datatype, r, schedule,
                            false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        free(tmpbuf);
        return res;
      }
    }

    if (comm_size == 1) {
      /* single node not in_place: copy data to recvbuf */
      res = NBC_Sched_copy ((void *)sendbuf, false, recvcounts[0], datatype,
                            recvbuf, false, recvcounts[0], datatype, schedule, false);
    } else {
      res = NBC_Sched_copy (lbuf, true, recvcounts[0], datatype, recvbuf, false,
                            recvcounts[0], datatype, schedule, false);
    }
  } else {
    res = NBC_Sched_recv (recvbuf, false, recvcounts[rank], datatype, 0, schedule, false);
  }

  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

  res = NBC_Sched_commit (schedule);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

  res = NBC_Schedule_request(schedule, comm, libnbc_module, persistent, request, tmpbuf);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

cleanup_and_return:
  return res;
}

/*
 * ompi_sum_counts: Returns sum of counts [lo, hi]
 *                  lo, hi in {0, 1, ..., nprocs_pof2 - 1}
 */
static int ompi_sum_counts(const int *counts, int *displs, int nprocs_rem, int lo, int hi)
{
    /* Adjust lo and hi for taking into account blocks of excluded processes */
    lo = (lo < nprocs_rem) ? lo * 2 : lo + nprocs_rem;
    hi = (hi < nprocs_rem) ? hi * 2 + 1 : hi + nprocs_rem;
    return displs[hi] + counts[hi] - displs[lo];
}

int ompi_coll_libnbc_ireduce_scatter (const void* sendbuf, void* recvbuf, const int *recvcounts, MPI_Datatype datatype,
                                      MPI_Op op, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                      struct mca_coll_base_module_2_3_0_t *module) {
    int res = nbc_reduce_scatter_init(sendbuf, recvbuf, recvcounts, datatype, op,
                                      comm, request, module, false);
    if (OPAL_LIKELY(OMPI_SUCCESS != res)) {
        return res;
    }
    res = NBC_Start(*(ompi_coll_libnbc_request_t **)request);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        NBC_Return_handle (*(ompi_coll_libnbc_request_t **)request);
        *request = &ompi_request_null.request;
        return res;
    }

    return OMPI_SUCCESS;
}
static int nbc_reduce_scatter_inter_init (const void* sendbuf, void* recvbuf, const int *recvcounts, MPI_Datatype datatype,
                                          MPI_Op op, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                          struct mca_coll_base_module_2_3_0_t *module, bool persistent) {
  int rank, res, count, lsize, rsize;
  MPI_Aint ext;
  ptrdiff_t gap, span, span_align;
  NBC_Schedule *schedule;
  void *tmpbuf = NULL;
  ompi_coll_libnbc_module_t *libnbc_module = (ompi_coll_libnbc_module_t*) module;

  rank = ompi_comm_rank (comm);
  lsize = ompi_comm_size(comm);
  rsize = ompi_comm_remote_size (comm);

  res = ompi_datatype_type_extent (datatype, &ext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  count = 0;
  for (int r = 0 ; r < lsize ; ++r) {
    count += recvcounts[r];
  }

  span = opal_datatype_span(&datatype->super, count, &gap);
  span_align = OPAL_ALIGN(span, datatype->super.align, ptrdiff_t);

  if (count > 0) {
    tmpbuf = malloc (span_align + span);
    if (OPAL_UNLIKELY(NULL == tmpbuf)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }
  }

  schedule = OBJ_NEW(NBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    free(tmpbuf);
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  /* send my data to the remote root */
  res = NBC_Sched_send(sendbuf, false, count, datatype, 0, schedule, false);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

  if (0 == rank) {
    char *lbuf, *rbuf;
    lbuf = (char *)(-gap);
    rbuf = (char *)(span_align-gap);
    res = NBC_Sched_recv (lbuf, true, count, datatype, 0, schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      free(tmpbuf);
      return res;
    }

    for (int peer = 1 ; peer < rsize ; ++peer) {
      char *tbuf;
      res = NBC_Sched_recv (rbuf, true, count, datatype, peer, schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        free(tmpbuf);
        return res;
      }

      res = NBC_Sched_op (lbuf, true, rbuf, true, count, datatype,
                          op, schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        free(tmpbuf);
        return res;
      }
      tbuf = lbuf; lbuf = rbuf; rbuf = tbuf;
    }

    /* do the local scatterv with the local communicator */
    res = NBC_Sched_copy (lbuf, true, recvcounts[0], datatype, recvbuf, false,
                          recvcounts[0], datatype, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      free(tmpbuf);
      return res;
    }
    for (int peer = 1, offset = recvcounts[0] * ext; peer < lsize ; ++peer) {
      res = NBC_Sched_local_send (lbuf + offset, true, recvcounts[peer], datatype, peer, schedule,
                                  false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        free(tmpbuf);
        return res;
      }

      offset += recvcounts[peer] * ext;
    }
  } else {
    /* receive my block */
    res = NBC_Sched_local_recv (recvbuf, false, recvcounts[rank], datatype, 0, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      free(tmpbuf);
      return res;
    }
  }

  res = NBC_Sched_commit (schedule);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

  res = NBC_Schedule_request(schedule, comm, libnbc_module, persistent, request, tmpbuf);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

  return OMPI_SUCCESS;
}

int ompi_coll_libnbc_ireduce_scatter_inter (const void* sendbuf, void* recvbuf, const int *recvcounts, MPI_Datatype datatype,
                                            MPI_Op op, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                            struct mca_coll_base_module_2_3_0_t *module) {
    int res = nbc_reduce_scatter_inter_init(sendbuf, recvbuf, recvcounts, datatype, op,
                                            comm, request, module, false);
    if (OPAL_LIKELY(OMPI_SUCCESS != res)) {
        return res;
    }
    res = NBC_Start(*(ompi_coll_libnbc_request_t **)request);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        NBC_Return_handle (*(ompi_coll_libnbc_request_t **)request);
        *request = &ompi_request_null.request;
        return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_libnbc_reduce_scatter_init(const void* sendbuf, void* recvbuf, const int *recvcounts, MPI_Datatype datatype,
                                         MPI_Op op, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                         struct mca_coll_base_module_2_3_0_t *module) {
    int res = nbc_reduce_scatter_init(sendbuf, recvbuf, recvcounts, datatype, op,
                                      comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_libnbc_reduce_scatter_inter_init(const void* sendbuf, void* recvbuf, const int *recvcounts, MPI_Datatype datatype,
                                               MPI_Op op, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                               struct mca_coll_base_module_2_3_0_t *module) {
    int res = nbc_reduce_scatter_inter_init(sendbuf, recvbuf, recvcounts, datatype, op,
                                            comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}
