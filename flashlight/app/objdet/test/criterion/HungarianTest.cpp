/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arrayfire.h>

#include "flashlight/lib/set/Hungarian.h"

#include <gtest/gtest.h>

using namespace fl::lib::set;

TEST(HungarianTest, DiagnalAssignments) {
  int M = 4; // Rows
  int N = 4; // Columns
  std::vector<float> costsVec(N * N);
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      costsVec[r * N + c] = (1 + r) * (1 + c);
    }
  }

  std::vector<int> expRowIdxs = {0, 1, 2, 3};
  std::vector<int> expColIdxs = {3, 2, 1, 0};
  std::vector<int> rowIdxs(N);
  std::vector<int> colIdxs(M);
  hungarian(costsVec.data(), rowIdxs.data(), colIdxs.data(), M, N);
  for (int r = 0; r < M; r++) {
    EXPECT_EQ(rowIdxs[r], expRowIdxs[r]) << "Assignment differs at index " << r;
  }
  for (int c = 0; c < N; c++) {
    EXPECT_EQ(rowIdxs[c], expRowIdxs[c]) << "Assignment differs at index " << c;
  }
}

TEST(HungarianTest, FullPipelineFromWiki) {
  int M = 3; // Rows
  int N = 3; // Columns
  // From https://en.wikipedia.org/wiki/Hungarian_algorithm
  std::vector<float> costsVec = {2, 3, 3, 3, 2, 3, 3, 3, 2};

  std::vector<int> expRowIdxs = {0, 1, 2};
  std::vector<int> expColIdxs = {0, 1, 2};

  std::vector<int> rowIdxs(N);
  std::vector<int> colIdxs(M);
  hungarian(costsVec.data(), rowIdxs.data(), colIdxs.data(), M, N);
  for (int r = 0; r < M; r++) {
    EXPECT_EQ(rowIdxs[r], expRowIdxs[r]) << "Assignment differs at index " << r;
  }
  for (int c = 0; c < N; c++) {
    EXPECT_EQ(rowIdxs[c], expRowIdxs[c]) << "Assignment differs at index " << c;
  }
}

TEST(HungarianTest, FullPipelineSimple1) {
  int M = 3; // Rows
  int N = 3; // Columns
  std::vector<float> costsVec = {
      1500,
      2000,
      2000,
      4000,
      6000,
      4000,
      4500,
      3500,
      2500,
  };

  std::vector<int> expAssignment = {0, 1, 0, 1, 0, 0, 0, 0, 1};
  std::vector<int> assignment(N * M);
  hungarian(costsVec.data(), assignment.data(), N, M);
  for (int c = 0; c < N; c++) {
    for (int r = 0; r < M; r++) {
      EXPECT_EQ(assignment[c * M + r], expAssignment[c * M + r])
          << "Assignment differs at row " << r << " and col " << c;
    }
  }
}

TEST(HungarianTest, FullPipelineSimple2) {
  int M = 3; // Rows
  int N = 3; // Columns
  std::vector<float> costsVec = {
      2500, 4000, 2000, 4000, 6000, 4000, 3500, 3500, 2500};

  std::vector<int> expAssignment = {0, 0, 1, 1, 0, 0, 0, 1, 0};
  std::vector<int> assignment(N * M);
  hungarian(costsVec.data(), assignment.data(), N, M);
  for (int c = 0; c < N; c++) {
    for (int r = 0; r < M; r++) {
      EXPECT_EQ(assignment[c * M + r], expAssignment[c * M + r])
          << "Assignment differs at row " << r << " and col " << c;
    }
  }
}

TEST(HungarianTest, FullPipelineSimple3) {
  int M = 3; // Rows
  int N = 3; // Columns
  std::vector<float> costsVec = {108, 150, 122, 125, 135, 148, 150, 175, 250};

  std::vector<int> expAssignment = {0, 0, 1, 0, 1, 0, 1, 0, 0};
  std::vector<int> assignment(N * M);
  hungarian(costsVec.data(), assignment.data(), N, M);
  for (int c = 0; c < N; c++) {
    for (int r = 0; r < M; r++) {
      EXPECT_EQ(assignment[c * M + r], expAssignment[c * M + r])
          << "Assignment differs at row " << r << " and col " << c;
    }
  }
}

TEST(HungarianTest, FullPipelineSize6) {
  int M = 6; // Rows
  int N = 6; // Columns
  std::vector<float> costsVec = {7, 9, 3, 7, 8, 4, 2, 6, 8, 9, 4, 2,
                                 1, 9, 3, 4, 7, 9, 9, 5, 1, 2, 4, 3,
                                 4, 5, 8, 2, 8, 1, 4, 2, 9, 3, 2, 9};

  std::vector<int> expAssignment = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                    1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                    0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0};
  std::vector<int> assignment(N * M);
  hungarian(costsVec.data(), assignment.data(), N, M);
  for (int c = 0; c < N; c++) {
    for (int r = 0; r < M; r++) {
      EXPECT_EQ(assignment[c * M + r], expAssignment[c * M + r])
          << "Assignment differs at row " << r << " and col " << c;
    }
  }
}
TEST(HungarianTest, 6x6Example2) {
  int M = 6; // Rows
  int N = 6; // Columns
  std::vector<float> costsVec = {7, 9, 3, 7, 8, 4, 2, 6, 8, 9, 4, 2,
                                 1, 9, 3, 4, 7, 9, 1, 3, 4, 8, 2, 7,
                                 4, 5, 8, 2, 8, 1, 4, 2, 9, 3, 2, 9};

  std::vector<int> expAssignment = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0};
  std::vector<int> assignment(N * M);
  hungarian(costsVec.data(), assignment.data(), N, M);
  for (int c = 0; c < N; c++) {
    for (int r = 0; r < M; r++) {
      EXPECT_EQ(assignment[c * M + r], expAssignment[c * M + r])
          << "Assignment differs at row " << r << " and col " << c;
    }
  }
}

// af::array cxcywh_to_xyxy(af::array bbox) {
// af::array transformed = af::constant(0, bbox.dims());
// for(int i = 0; i < bbox.dims(1); i++) {
// float x_c = bbox(i, 0);
// float y_c = bbox(i, 1);
// float w = bbox(i, 2);
// float h = bbox(i, 3);
// transformed(0, i) = x_c - 0.5 * w;
// transformed(1, i) = y_c - 0.5 * h;
// transformed(2, i) = x_c + 0.5 * w;
// transformed(3, i) = y_c + 0.5 * h;
//}

//}
TEST(HungarianTest, NonSquare2) {
  int M = 1; // Rows
  int N = 2; // Columns
  std::vector<float> costsVec = {0, 0.5};

  std::vector<int> expRowIdxs = {0};
  std::vector<int> expColIdxs = {1};

  const int num_indices = std::min(N, M);
  std::vector<int> rowIdxs(num_indices, -1);
  std::vector<int> colIdxs(num_indices, -1);
  hungarian(costsVec.data(), rowIdxs.data(), colIdxs.data(), M, N);
  for (int i = 0; i < num_indices; i++) {
    EXPECT_EQ(rowIdxs[i], expRowIdxs[i]) << "Assignment differs at index " << i;
    EXPECT_EQ(colIdxs[i], colIdxs[i]) << "Assignment differs at index " << i;
  }
}

TEST(HungarianTest, NonSquare) {
  int M = 1; // Rows
  int N = 2; // Columns
  std::vector<float> costsVec = {0.5, 0};

  std::vector<int> expRowIdxs = {0};
  std::vector<int> expColIdxs = {0};

  const int num_indices = std::min(N, M);
  std::vector<int> rowIdxs(num_indices, -1);
  std::vector<int> colIdxs(num_indices, -1);
  hungarian(costsVec.data(), rowIdxs.data(), colIdxs.data(), M, N);
  for (int i = 0; i < num_indices; i++) {
    EXPECT_EQ(rowIdxs[i], expRowIdxs[i]) << "Assignment differs at index " << i;
    EXPECT_EQ(colIdxs[i], colIdxs[i]) << "Assignment differs at index " << i;
  }
}

TEST(HungarianTest, NonSquare3) {
  int M = 2; // Rows
  int N = 3; // Columns
  std::vector<float> costsVec = {
      0,
      0.5,
      0.5,
      2,
      2,
      3,
  };

  std::vector<int> expRowIdxs = {
      0,
      1,
  };
  std::vector<int> expColIdxs = {1, 0};

  const int num_indices = std::min(N, M);
  std::vector<int> rowIdxs(num_indices, -1);
  std::vector<int> colIdxs(num_indices, -1);
  hungarian(costsVec.data(), rowIdxs.data(), colIdxs.data(), M, N);
  for (int i = 0; i < num_indices; i++) {
    EXPECT_EQ(rowIdxs[i], expRowIdxs[i]) << "Assignment differs at index " << i;
    EXPECT_EQ(colIdxs[i], colIdxs[i]) << "Assignment differs at index " << i;
  }
}
