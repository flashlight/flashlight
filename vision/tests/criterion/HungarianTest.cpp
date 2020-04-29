#include <arrayfire.h>

#include "vision/criterion/Hungarian.cpp"

#include <gtest/gtest.h>


TEST(HungarianTest, Step1) {

  int N = 3;
  std::vector<float> costsVec(N * N);
  for(int r = 0; r < N; r++) {
    for(int c = 0; c < N; c++) {
      costsVec[r * N + c] = (1 + r) * (1 + c);
    }
  }
  std::vector<float> expected = {
    0, 0, 0, 1, 2, 3, 2, 4, 6
  };
  step_one(costsVec.data(), N, N);
  for (int i = 0; i < costsVec.size(); ++i) {
    EXPECT_EQ(costsVec[i], expected[i]) << "Vectors x and y differ at index " << i;
  }
}


TEST(HungarianTest, Step2) {

  int N = 3;
  std::vector<float> costsVec(N * N);
  for(int r = 0; r < N; r++) {
    for(int c = 0; c < N; c++) {
      costsVec[r * N + c] = (1 + r) * (1 + c);
    }
  }

  std::vector<int> marks(N * N);
  std::vector<int> rowCover(N);
  std::vector<int> colCover(N);
  std::vector<int> expectedMarks = {
    1, 0, 0, 0, 0, 0, 0, 0, 0
  };
  step_one(costsVec.data(), N, N);
  step_two(costsVec.data(), marks.data(), rowCover.data(), colCover.data(), N, N);
  for (int i = 0; i < costsVec.size(); ++i) {
    EXPECT_EQ(marks[i], expectedMarks[i]) << "Vectors x and y differ at index " << i;
  }
}



TEST(HungarianTest, Step3) {
  int N = 3;
  std::vector<float> costsVec(N * N);
  for(int r = 0; r < N; r++) {
    for(int c = 0; c < N; c++) {
      costsVec[r * N + c] = (1 + r) * (1 + c);
    }
  }

  std::vector<int> marks(N * N);
  std::vector<int> rowCover(N);
  std::vector<int> colCover(N);
  std::vector<int> expectedColCover = {
    1, 0, 0
  };
  step_one(costsVec.data(), N, N);
  step_two(costsVec.data(), marks.data(), colCover.data(), rowCover.data(), N, N);
  int nextStep = step_three(marks.data(), colCover.data(), rowCover.data(), N, N);
  for(int c = 0; c < N; c++) {
    EXPECT_EQ(colCover[c], expectedColCover[c]) << "column coverage differs at index " << c;
  }
  EXPECT_EQ(nextStep, 4);
}

TEST(HungarianTest, Step4) {
  int N = 3;
  std::vector<float> costsVec(N * N);
  for(int r = 0; r < N; r++) {
    for(int c = 0; c < N; c++) {
      costsVec[r * N + c] = (1 + r) * (1 + c);
    }
  }
  int firstPathRow, firstPathCol;

  std::vector<int> marks(N * N);
  std::vector<int> rowCover(N);
  std::vector<int> colCover(N);
  std::vector<int> expectedColCover = {
    1, 0, 0
  };
  step_one(costsVec.data(), N, N);
  step_two(costsVec.data(), marks.data(), colCover.data(), rowCover.data(), N, N);
  step_three(marks.data(), colCover.data(), rowCover.data(), N, N);
  int nextStep = step_four(costsVec.data(), marks.data(), colCover.data(), rowCover.data(), N, N, &firstPathRow, &firstPathCol);
  EXPECT_EQ(nextStep, 6);
}

TEST(HungarianTest, Step6) {
  int N = 3;
  std::vector<float> costsVec(N * N);
  for(int r = 0; r < N; r++) {
    for(int c = 0; c < N; c++) {
      costsVec[r * N + c] = (1 + r) * (1 + c);
    }
  }

  std::vector<int> marks(N * N);
  std::vector<int> rowCover(N);
  std::vector<int> colCover(N);
  std::vector<int> expectedCosts = {
    0, 0, 0, 0, 1, 2, 1, 3, 5
  };
  int firstPathRow, firstPathCol;
  step_one(costsVec.data(), N, N);
  step_two(costsVec.data(), marks.data(), colCover.data(), rowCover.data(), N, N);
  step_three(marks.data(), colCover.data(), rowCover.data(), N, N);
  step_four(costsVec.data(), marks.data(), colCover.data(), rowCover.data(), N, N, &firstPathRow, &firstPathCol);
  int nextStep = step_six(costsVec.data(), marks.data(), colCover.data(), rowCover.data(), N, N);
  for(int i = 0; i < N * N; i++) {
    EXPECT_EQ(expectedCosts[i], costsVec[i]) << "Costs differ at index " << i;
  }
  EXPECT_EQ(nextStep, 4);
}

TEST(HungarianTest, Step123464) {
  int N = 3;
  std::vector<float> costsVec(N * N);
  for(int r = 0; r < N; r++) {
    for(int c = 0; c < N; c++) {
      costsVec[r * N + c] = (1 + r) * (1 + c);
    }
  }

  std::vector<int> marks(N * N);
  std::vector<int> rowCover(N);
  std::vector<int> colCover(N);
  int firstPathRow, firstPathCol;
  std::vector<int> expColCover = {
    0, 0, 0
  };
  std::vector<int> expRowCover = {
    1, 0, 0
  };
  std::vector<int> expMarks = {
    1, 2, 0, 2, 0, 0, 0, 0, 0
  };
  step_one(costsVec.data(), N, N);
  step_two(costsVec.data(), marks.data(), colCover.data(), rowCover.data(), N, N);
  step_three(marks.data(), colCover.data(), rowCover.data(), N, N);
  step_four(costsVec.data(), marks.data(), colCover.data(), rowCover.data(), N, N, &firstPathRow, &firstPathCol);
  int nextStep = step_six(costsVec.data(), marks.data(), colCover.data(), rowCover.data(), N, N);
  EXPECT_EQ(nextStep, 4);
  nextStep = step_four(costsVec.data(), marks.data(), colCover.data(), rowCover.data(), N, N, 
      &firstPathRow, &firstPathCol);
  for(int i = 0; i < N; i++) {
    EXPECT_EQ(expRowCover[i], rowCover[i]) << "Row cover differ at  index " << i;
  }
  for(int i = 0; i < N; i++) {
    EXPECT_EQ(expColCover[i], colCover[i]) << "Col cover differ at index " << i;
  }
  for(int i = 0; i < N * N; i++) {
    EXPECT_EQ(expMarks[i], marks[i]) << "Col cover differ at index " << i;
  }
  EXPECT_EQ(firstPathRow, 1);
  EXPECT_EQ(firstPathCol, 0);
  EXPECT_EQ(nextStep, 5);
}

TEST(HungarianTest, Step1234645) {
  int N = 3;
  std::vector<float> costsVec(N * N);
  for(int r = 0; r < N; r++) {
    for(int c = 0; c < N; c++) {
      costsVec[r * N + c] = (1 + r) * (1 + c);
    }
  }

  std::vector<int> marks(N * N);
  std::vector<int> rowCover(N);
  std::vector<int> colCover(N);
  int firstPathRow, firstPathCol;
  std::vector<int> expColCover = {
    0, 0, 0
  };
  std::vector<int> expRowCover = {
    0, 0, 0
  };
  std::vector<int> expMarks = {
    0, 1, 0, 1, 0, 0, 0, 0, 0
  };
  step_one(costsVec.data(), N, N);
  step_two(costsVec.data(), marks.data(), colCover.data(), rowCover.data(), N, N);
  step_three(marks.data(), colCover.data(), rowCover.data(), N, N);
  step_four(costsVec.data(), marks.data(), colCover.data(), rowCover.data(), N, N, &firstPathRow, &firstPathCol);
  int nextStep = step_six(costsVec.data(), marks.data(), colCover.data(), rowCover.data(), N, N);
  EXPECT_EQ(nextStep, 4);
  nextStep = step_four(costsVec.data(), marks.data(), colCover.data(), rowCover.data(), N, N, 
      &firstPathRow, &firstPathCol);

  EXPECT_EQ(firstPathRow, 1);
  EXPECT_EQ(firstPathCol, 0);
  EXPECT_EQ(nextStep, 5);
  std::vector<int> paths(N * N * 2);
  step_five(costsVec.data(), marks.data(), colCover.data(), rowCover.data(), paths.data(), firstPathRow, firstPathCol, N, N);
  for(int i = 0; i < N; i++) {
    EXPECT_EQ(expRowCover[i], rowCover[i]) << "Row cover differ at  index " << i;
  }
  for(int i = 0; i < N; i++) {
    EXPECT_EQ(expColCover[i], colCover[i]) << "Col cover differ at index " << i;
  }
  for(int i = 0; i < N * N; i++) {
    EXPECT_EQ(expMarks[i], marks[i]) << "Col cover differ at index " << i;
  }
}

TEST(HungarianTest, FullRunDiagnalAssignments) {
  int M = 3; // Rows
  int N = 3; // Columns
  std::vector<float> costsVec(N * N);
  for(int r = 0; r < M; r++) {
    for(int c = 0; c < N; c++) {
      costsVec[r * N + c] = (1 + r) * (1 + c);
    }
  }
  std::vector<int> expAssignment = {
    0, 0, 1, 0, 1, 0, 1, 0, 0, 0
  };
  std::vector<int> assignment(N * M);
  hungarian(costsVec.data(), assignment.data(), N, M);
  for(int c = 0; c < N; c++) {
    for(int r = 0; r < M; r++) {
      EXPECT_EQ(assignment[c * M + r], expAssignment[c * M + r]) 
        << "Assignment differs at row " << r << " and col " << c;
    }
  }

}

TEST(HungarianTest, FullPipelineFromWiki) {
  int M = 3; // Rows
  int N = 3; // Columns
  // From https://en.wikipedia.org/wiki/Hungarian_algorithm
  std::vector<float> costsVec = {
    2, 3, 3,
    3, 2, 3, 
    3, 3, 2
  };
  
  std::vector<int> expAssignment = {
    1, 0, 0, 0, 1, 0, 0, 0, 1
  };
  std::vector<int> assignment(N * M);
  hungarian(costsVec.data(), assignment.data(), N, M);
  for(int c = 0; c < N; c++) {
    for(int r = 0; r < M; r++) {
      EXPECT_EQ(assignment[c * M + r], expAssignment[c * M + r]) 
        << "Assignment differs at row " << r << " and col " << c;
    }
  }

}

TEST(HungarianTest, FullPipelineSimple1) {
  int M = 3; // Rows
  int N = 3; // Columns
  std::vector<float> costsVec = {
    1500, 2000, 2000,
    4000, 6000, 4000,
    4500, 3500, 2500,
  };
  
  std::vector<int> expAssignment = {
    0, 1, 0,
    1, 0, 0, 
    0, 0, 1
  };
  std::vector<int> assignment(N * M);
  hungarian(costsVec.data(), assignment.data(), N, M);
  for(int c = 0; c < N; c++) {
    for(int r = 0; r < M; r++) {
      EXPECT_EQ(assignment[c * M + r], expAssignment[c * M + r]) 
        << "Assignment differs at row " << r << " and col " << c;
    }
  }

}

TEST(HungarianTest, FullPipelineSimple2) {
  int M = 3; // Rows
  int N = 3; // Columns
  std::vector<float> costsVec = {
    2500, 4000, 2000,
    4000, 6000, 4000,
    3500, 3500, 2500
  };
  
  std::vector<int> expAssignment = {
    0, 0, 1,
    1, 0, 0,
    0, 1, 0
  };
  std::vector<int> assignment(N * M);
  hungarian(costsVec.data(), assignment.data(), N, M);
  for(int c = 0; c < N; c++) {
    for(int r = 0; r < M; r++) {
      EXPECT_EQ(assignment[c * M + r], expAssignment[c * M + r]) 
        << "Assignment differs at row " << r << " and col " << c;
    }
  }
}

TEST(HungarianTest, FullPipelineSimple3) {
  int M = 3; // Rows
  int N = 3; // Columns
  std::vector<float> costsVec = {
    108, 150, 122, 
    125, 135, 148, 
    150, 175, 250
  };
  
  std::vector<int> expAssignment = {
    0, 0, 1,
    0, 1, 0,
    1, 0, 0
  };
  std::vector<int> assignment(N * M);
  hungarian(costsVec.data(), assignment.data(), N, M);
  for(int c = 0; c < N; c++) {
    for(int r = 0; r < M; r++) {
      EXPECT_EQ(assignment[c * M + r], expAssignment[c * M + r]) 
        << "Assignment differs at row " << r << " and col " << c;
    }
  }
}

TEST(HungarianTest, FullPipelineSize6) {
  int M = 6; // Rows
  int N = 6; // Columns
  std::vector<float> costsVec = {
    7, 9, 3, 7, 8, 4,
    2, 6, 8, 9, 4, 2,
    1, 9, 3, 4, 7, 9,
    9, 5, 1, 2, 4, 3,
    4, 5, 8, 2, 8, 1,
    4, 2, 9, 3, 2, 9
  };
  
  std::vector<int> expAssignment = {
    0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 1, 0,
    1, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 1,
    0, 1, 0, 0, 0, 0
  };
  std::vector<int> assignment(N * M);
  hungarian(costsVec.data(), assignment.data(), N, M);
  for(int c = 0; c < N; c++) {
    for(int r = 0; r < M; r++) {
      EXPECT_EQ(assignment[c * M + r], expAssignment[c * M + r]) 
        << "Assignment differs at row " << r << " and col " << c;
    }
  }
}
TEST(HungarianTest, 6x6Example2) {
  int M = 6; // Rows
  int N = 6; // Columns
  std::vector<float> costsVec = {
    7, 9, 3, 7, 8, 4,
    2, 6, 8, 9, 4, 2, 
    1, 9, 3, 4, 7, 9,
    1, 3, 4, 8, 2, 7,
    4, 5, 8, 2, 8, 1,
    4, 2, 9, 3, 2, 9
  };
  
  std::vector<int> expAssignment = {
    0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 1, 0, 0,
    0, 1, 0, 0, 0, 0
  };
  std::vector<int> assignment(N * M);
  hungarian(costsVec.data(), assignment.data(), N, M);
  for(int c = 0; c < N; c++) {
    for(int r = 0; r < M; r++) {
      EXPECT_EQ(assignment[c * M + r], expAssignment[c * M + r]) 
        << "Assignment differs at row " << r << " and col " << c;
    }
  }
}
