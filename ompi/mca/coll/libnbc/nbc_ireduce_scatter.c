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
    struct ompi_communicator_t *comm,
    int rank, int comm_size, NBC_Schedule *schedule,
    const void *sendbuf, void *recvbuf, const int *recvcounts, MPI_Datatype datatype,
    int count, MPI_Op op, ompi_coll_libnbc_module_t *libnbc_module,
    ompi_request_t ** request, bool persistent);

static inline int reduce_scatter_butterfly(
    const void *sendbuf, void *recvbuf, const int *recvcounts, MPI_Datatype dtype,
    MPI_Op op, struct ompi_communicator_t *comm,
    ompi_coll_libnbc_module_t *module,
    ompi_request_t ** request, bool persistent);

static int ompi_sum_counts(const int *counts, int *displs, int nprocs_rem, int lo, int hi);

/* an reduce_csttare schedule can not be cached easily because the contents
 * ot the recvcounts array may change, so a comparison of the address
 * would not be sufficient ... we simply do not cache it */

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

static int nbc_reduce_scatter_init(const void* sendbuf, void* recvbuf, const int *recvcounts, MPI_Datatype datatype,
                                   MPI_Op op, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                   struct mca_coll_base_module_2_3_0_t *module, bool persistent) {
  int rank, p, res, count;
  char inplace;
  NBC_Schedule *schedule;
  ompi_coll_libnbc_module_t *libnbc_module = (ompi_coll_libnbc_module_t*) module;

  enum { NBC_REDUCE_SCAT_PAIRWISE_EXCHANGE, NBC_REDUCE_SCAT_BUTTERFLY } alg;

  NBC_IN_PLACE(sendbuf, recvbuf, inplace);

  rank = ompi_comm_rank (comm);
  p = ompi_comm_size (comm);

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

  // res = ompi_datatype_type_extent (datatype, &ext);
  // if (MPI_SUCCESS != res) {
  //   NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
  //   return res;
  // }

  count = 0;
  for (int r = 0 ; r < p ; ++r) {
    count += recvcounts[r];
  }

  if ((1 == p && (!persistent || inplace)) || 0 == count) {
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
      if (rank == 0) printf("Reduce scatter algorithm is: pairwise_exchange\n");
      res = reduce_scatter_pairwise_exchange(comm, rank, p, schedule, sendbuf,
                                             recvbuf, recvcounts, datatype, count, op,
                                             libnbc_module, request, persistent);
      break;
    case NBC_REDUCE_SCAT_BUTTERFLY:
      if (rank == 0) printf("Reduce scatter algorithm is: butterfly\n");
      res = reduce_scatter_butterfly(sendbuf, recvbuf, recvcounts,
                                    datatype, op, comm, libnbc_module,
                                    request, persistent);
      break;
  }

  printf("Reduce scatter: after algorithm selection\n");

#pragma region 
  // maxr = (int) ceil ((log((double) p) / LOG2));

  // span = opal_datatype_span(&datatype->super, count, &gap);
  // span_align = OPAL_ALIGN(span, datatype->super.align, ptrdiff_t);
  // tmpbuf = malloc (span_align + span);
  // if (OPAL_UNLIKELY(NULL == tmpbuf)) {
  //   return OMPI_ERR_OUT_OF_RESOURCE;
  // }

  // rbuf = (char *)(-gap);
  // lbuf = (char *)(span_align - gap);

  // schedule = OBJ_NEW(NBC_Schedule);
  // if (OPAL_UNLIKELY(NULL == schedule)) {
  //   free(tmpbuf);
  //   return OMPI_ERR_OUT_OF_RESOURCE;
  // }

  // for (int r = 1, firstred = 1 ; r <= maxr ; ++r) {
  //   if ((rank % (1 << r)) == 0) {
  //     /* we have to receive this round */
  //     peer = rank + (1 << (r - 1));
  //     if (peer < p) {
  //       /* we have to wait until we have the data */
  //       res = NBC_Sched_recv(rbuf, true, count, datatype, peer, schedule, true);
  //       if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
  //         OBJ_RELEASE(schedule);
  //         free(tmpbuf);
  //         return res;
  //       }

  //       /* this cannot be done until tmpbuf is unused :-( so barrier after the op */
  //       if (firstred) {
  //         /* take reduce data from the sendbuf in the first round -> save copy */
  //         res = NBC_Sched_op (sendbuf, false, rbuf, true, count, datatype, op, schedule, true);
  //         firstred = 0;
  //       } else {
  //         /* perform the reduce in my local buffer */
  //         res = NBC_Sched_op (lbuf, true, rbuf, true, count, datatype, op, schedule, true);
  //       }

  //       if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
  //         OBJ_RELEASE(schedule);
  //         free(tmpbuf);
  //         return res;
  //       }
  //       /* swap left and right buffers */
  //       buf = rbuf; rbuf = lbuf ; lbuf = buf;
  //     }
  //   } else {
  //     /* we have to send this round */
  //     peer = rank - (1 << (r - 1));
  //     if (firstred) {
  //       /* we have to send the senbuf */
  //       res = NBC_Sched_send (sendbuf, false, count, datatype, peer, schedule, false);
  //     } else {
  //       /* we send an already reduced value from lbuf */
  //       res = NBC_Sched_send (lbuf, true, count, datatype, peer, schedule, false);
  //     }
  //     if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
  //       OBJ_RELEASE(schedule);
  //       free(tmpbuf);
  //       return res;
  //     }

  //     /* leave the game */
  //     break;
  //   }
  // }  

  // res = NBC_Sched_barrier(schedule);
  // if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
  //   OBJ_RELEASE(schedule);
  //   free(tmpbuf);
  //   return res;
  // }

  // /* rank 0 is root and sends - all others receive */
  // if (rank == 0) {
  //   for (long int r = 1, offset = 0 ; r < p ; ++r) {
  //     offset += recvcounts[r-1];
  //     sbuf = lbuf + (offset*ext);
  //     /* root sends the right buffer to the right receiver */
  //     res = NBC_Sched_send (sbuf, true, recvcounts[r], datatype, r, schedule,
  //                           false);
  //     if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
  //       OBJ_RELEASE(schedule);
  //       free(tmpbuf);
  //       return res;
  //     }
  //   }

  //   if (p == 1) {
  //     /* single node not in_place: copy data to recvbuf */
  //     res = NBC_Sched_copy ((void *)sendbuf, false, recvcounts[0], datatype,
  //                           recvbuf, false, recvcounts[0], datatype, schedule, false);
  //   } else {
  //     res = NBC_Sched_copy (lbuf, true, recvcounts[0], datatype, recvbuf, false,
  //                           recvcounts[0], datatype, schedule, false);
  //   }
  // } else {
  //   res = NBC_Sched_recv (recvbuf, false, recvcounts[rank], datatype, 0, schedule, false);
  // }

  // if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
  //   OBJ_RELEASE(schedule);
  //   free(tmpbuf);
  //   return res;
  // }

  // res = NBC_Sched_commit (schedule);
  // if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
  //   OBJ_RELEASE(schedule);
  //   free(tmpbuf);
  //   return res;
  // }

  // res = NBC_Schedule_request(schedule, comm, libnbc_module, persistent, request, tmpbuf);
  // if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
  //   OBJ_RELEASE(schedule);
  //   free(tmpbuf);
  //   return res;
  // }
  #pragma endregion

  return OMPI_SUCCESS;
}

static inline int reduce_scatter_butterfly(
  const void *sendbuf, void *recvbuf, const int *recvcounts, MPI_Datatype dtype,
  MPI_Op op, struct ompi_communicator_t *comm,
  ompi_coll_libnbc_module_t *module,
  ompi_request_t ** request, bool persistent)
{
  NBC_Schedule *schedule;
  char *tmpbuf[2] = {NULL, NULL}, *psend, *precv;
  int *displs = NULL, index;
  ptrdiff_t span, gap, totalcount, extent;
  int err = MPI_SUCCESS;
  int comm_size = ompi_comm_size(comm);
  int rank = ompi_comm_rank(comm);

  // OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
  //               "coll:base:reduce_scatter_intra_butterfly: rank %d/%d",
  //               rank, comm_size));
  if (comm_size < 2)
      return MPI_SUCCESS;

  printf("Rank: %d: before line %d\n\n", rank, 277);
  schedule = OBJ_NEW(NBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    err = OMPI_ERR_OUT_OF_RESOURCE;
    goto cleanup_and_return;
  }
  printf("Rank: %d: after line %d\n", rank, 277);

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

  ompi_datatype_type_extent(dtype, &extent);
  span = opal_datatype_span(&dtype->super, totalcount, &gap);
  tmpbuf[0] = malloc(span);
  tmpbuf[1] = malloc(span);
  if (NULL == tmpbuf[0] || NULL == tmpbuf[1]) {
      err = OMPI_ERR_OUT_OF_RESOURCE;
      goto cleanup_and_return;
  }
  psend = tmpbuf[0] - gap;
  precv = tmpbuf[1] - gap;

  if (sendbuf != MPI_IN_PLACE) {
    printf("Rank: %d: before line %d\n\n", rank, 309);

    err = NBC_Copy(sendbuf, totalcount, dtype, psend, totalcount, dtype, comm);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
      goto cleanup_and_return;
    }

    printf("Rank: %d: after line %d\n", rank, 309);
  } else {
    printf("Rank: %d: before line %d\n", rank, 318);

    err = NBC_Copy(recvbuf, totalcount, dtype, psend, totalcount, dtype, comm);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
      goto cleanup_and_return;
    }

    printf("Rank: %d: after line %d\n", rank, 318);
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
          printf("Rank: %d: before line %d\n", rank, 351);

          err = NBC_Sched_send(psend, false, totalcount, dtype, rank + 1, schedule, false);
          if (OMPI_SUCCESS != err) { goto cleanup_and_return; }

          printf("Rank: %d: after line %d\n", rank, 351);

          /* This process does not participate in the rest of the algorithm */
          vrank = -1;
      } else {
          /* Odd process */
          printf("Rank: %d: before line %d\n", rank, 362);

          err = NBC_Sched_recv(precv, true, totalcount, dtype, rank - 1, schedule, false);
          if (OMPI_SUCCESS != err) { goto cleanup_and_return; }

          printf("Rank: %d: after line %d\n", rank, 362);
          printf("Rank: %d: before line %d\n", rank, 368);

          err = NBC_Sched_op(precv, true, psend, true, totalcount,
                             dtype, op, schedule, true);
          if (OMPI_SUCCESS != err) { goto cleanup_and_return; }
          printf("Rank: %d: after line %d\n", rank, 368);

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

        printf("Rank: %d: before line %d\n", rank, 421);

        err = NBC_Sched_send(psend + (ptrdiff_t)sdispl * extent, false, send_count,
                             dtype, peer, schedule, false);
        if (MPI_SUCCESS != err) { goto cleanup_and_return; }

        printf("Rank: %d: after line %d\n", rank, 421);
        printf("Rank: %d: before line %d\n", rank, 428);

        err = NBC_Sched_recv(precv + (ptrdiff_t)sdispl * extent, false, recv_count,
                             dtype, peer, schedule, false);
        if (MPI_SUCCESS != err) { goto cleanup_and_return; }

        printf("Rank: %d: after line %d\n", rank, 428);         

        if (vrank < vpeer) {
          /* precv = psend <op> precv */
          printf("Rank: %d: before line %d\n", rank, 438);

          err = NBC_Sched_op(psend + (ptrdiff_t)rdispl * extent, false,
                        precv + (ptrdiff_t)rdispl * extent, false,
                        recv_count, dtype, op, schedule, true);

          if (MPI_SUCCESS != err) { goto cleanup_and_return; }

          printf("Rank: %d: after line %d\n", rank, 438);

          char *p = psend;
          psend = precv;
          precv = p;
        } else {
          /* psend = precv <op> psend */
          printf("Rank: %d: before line %d\n", rank, 453);

          err = NBC_Sched_op(precv + (ptrdiff_t)rdispl * extent, false,
                             psend + (ptrdiff_t)rdispl * extent, false,
                             recv_count, dtype, op, schedule, true);

          if (MPI_SUCCESS != err) { goto cleanup_and_return; }

          printf("Rank: %d: after line %d\n", rank, 453);
        }
        send_index = recv_index;
      }

      printf("Rank: %d: before line %d\n", rank, 466);

      err = NBC_Sched_barrier(schedule);
      if (MPI_SUCCESS != err) { goto cleanup_and_return; }

      printf("Rank: %d: after line %d\n", rank, 466);

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
          printf("Rank: %d: before line %d\n", rank, 486);

          err = NBC_Sched_send(psend + (ptrdiff_t)displs[index] * extent, false,
                               recvcounts[index], dtype, peer - 1,
                               schedule, false);
          if (MPI_SUCCESS != err) { goto cleanup_and_return; }

          printf("Rank: %d: after line %d\n", rank, 486);
      }

      /* If process has two blocks, then send the second block (own block) */
      if (vpeer < nprocs_rem)
          index++;
      if (vpeer != vrank) {
        printf("Rank: %d: before line %d\n", rank, 500);

        err = NBC_Sched_send(psend + (ptrdiff_t)displs[index] * extent, false, recvcounts[index],
                             dtype, peer, schedule, false);
        if (MPI_SUCCESS != err) { goto cleanup_and_return; }

        printf("Rank: %d: after line %d\n", rank, 500);
        printf("Rank: %d: before line %d\n", rank, 507);

        err = NBC_Sched_recv(recvbuf, false, recvcounts[rank],
                             dtype, peer, schedule, false);
        if (MPI_SUCCESS != err) { goto cleanup_and_return; }

        printf("Rank: %d: after line %d\n", rank, 507);
      } else {
        printf("Rank: %d: before line %d\n", rank, 515);

        err = NBC_Sched_copy(psend + (ptrdiff_t)displs[rank] * extent, false,
                             recvcounts[rank], dtype,
                             recvbuf, false,
                             recvcounts[rank], dtype,
                             schedule, false);

        if (MPI_SUCCESS != err) { goto cleanup_and_return; }

        printf("Rank: %d: after line %d\n", rank, 515);
      }

  } else {
      /* Excluded process: receive result */
      int vpeer = ompi_mirror_perm((rank + 1) / 2, log2_size);
      int peer = (vpeer < nprocs_rem) ? vpeer * 2 + 1 : vpeer + nprocs_rem;

      printf("Rank: %d: before line %d\n", rank, 533);

      err = NBC_Sched_recv(recvbuf, false, recvcounts[rank],
                           dtype, peer, schedule, false);
                           
      if (OMPI_SUCCESS != err) { goto cleanup_and_return; }

      printf("Rank: %d: after line %d\n", rank, 533);
  }

  printf("Rank: %d: before line %d\n", rank, 543);

  err = NBC_Sched_commit (schedule);
  if (OMPI_SUCCESS != err) { goto cleanup_and_return; }

  printf("Rank: %d: after line %d\n", rank, 543);
  printf("Rank: %d: before line %d\n", rank, 549);

  err = NBC_Schedule_request(schedule, comm, module, persistent, request, tmpbuf);
  if (OMPI_SUCCESS != err) { goto cleanup_and_return; }

  printf("Rank: %d: after line %d\n", rank, 549);

cleanup_and_return:
  printf("Rank: %d: cleanup and return\n", rank);
  if (displs)
      free(displs);
  if (tmpbuf[0])
      free(tmpbuf[0]);
  if (tmpbuf[1])
      free(tmpbuf[1]);

  OBJ_RELEASE(schedule);
  return err;
}

static inline int reduce_scatter_pairwise_exchange(
    struct ompi_communicator_t *comm,
    int rank, int comm_size, NBC_Schedule *schedule, 
    const void *sendbuf, void *recvbuf, const int *recvcounts, MPI_Datatype datatype,
    int count, MPI_Op op, ompi_coll_libnbc_module_t *libnbc_module,
    ompi_request_t ** request, bool persistent) 
{
  MPI_Aint ext;
  int res, maxr, peer;
  ptrdiff_t gap, span, span_align;
  void *tmpbuf;
  char *rbuf, *lbuf, *buf, *sbuf;
 
  res = ompi_datatype_type_extent (datatype, &ext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  res = OMPI_SUCCESS;
  maxr = (int) ceil ((log((double) comm_size) / LOG2));

  span = opal_datatype_span(&datatype->super, count, &gap);
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
        res = NBC_Sched_recv(rbuf, true, count, datatype, peer, schedule, true);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          OBJ_RELEASE(schedule);
          free(tmpbuf);
          return res;
        }

        /* this cannot be done until tmpbuf is unused :-( so barrier after the op */
        if (firstred) {
          /* take reduce data from the sendbuf in the first round -> save copy */
          res = NBC_Sched_op (sendbuf, false, rbuf, true, count, datatype, op, schedule, true);
          firstred = 0;
        } else {
          /* perform the reduce in my local buffer */
          res = NBC_Sched_op (lbuf, true, rbuf, true, count, datatype, op, schedule, true);
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
        res = NBC_Sched_send (sendbuf, false, count, datatype, peer, schedule, false);
      } else {
        /* we send an already reduced value from lbuf */
        res = NBC_Sched_send (lbuf, true, count, datatype, peer, schedule, false);
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
