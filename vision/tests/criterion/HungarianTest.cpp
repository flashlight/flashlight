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
  //af_print(af::array(af::dim4(3, 3), costsVec.data()));
  step_one(costsVec.data(), N, N);
  //af_print(af::array(af::dim4(3, 3), costsVec.data()));
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
  //af_print(af::array(af::dim4(3, 3), costsVec.data()));
  step_one(costsVec.data(), N, N);
  step_two(costsVec.data(), marks.data(), rowCover.data(), colCover.data(), N, N);
  //af_print(af::array(af::dim4(3, 3), costsVec.data()));
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
  af_print(af::array(af::dim4(N, N), marks.data()).T());
  af_print(af::array(af::dim4(2, N * N), paths.data()));
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

//TEST(HungarianTest, MinLinesCoverage) {

  //std::vector<float> costsVec = {
    //0., 1., 1., 1.,
    //1., 1., 1., 0.,
    //0., 1., 1., 1,
    //1., 0., 0., 1.,
  //};

  ////std::vector<float> costsVec = {
    ////1, 2, 3, 4, 
    ////5, 6, 7, 8, 
    ////9, 10, 11, 12, 
    ////13, 14, 15, 16,
  ////};
  //af::array costs(af::dim4(4, 4), costsVec.data());
  //af::array masks = af::constant(0.0f, af::dim4(4, 4));
  //af::array assignments = af::constant(0, af::dim4(4, 4));
  //af_print(costs > 0);
  //af_print(costs(costs > 0));
  //costs = costs.T();
  //af::array min_vals, min_idxs;
  ////topk(min_vals, min_idxs, costs, 1, AF_TOPK_MAX);
  //for(int i = 0; i < 4; i++) {
    //af_print(costs);
    //af_print(assignments);
    //af_print(masks);
    //af_print(costs(i, af::span));
    //af::min(min_vals, min_idxs, (masks + costs)(i, af::span), 1);    
    //if (min_vals.scalar<float>() == 0.0f) {
      //af_print(costs(i, af::span) == 0);
      //af_print(costs(af::span, min_idxs) == 0);
      //masks(af::span, min_idxs) = 1 - costs(af::span, min_idxs);
      //masks(i, af::span) = 1 - costs(i, af::span);
      //assignments(i, min_idxs) = 1;
    //}
  //}
  //masks = masks - assignments;
  //af::array rowLines = af::constant<bool>(false, 4, b8);
  //af::array colLines = af::constant<bool>(false, 4, b8);
  //af_print(assignments);
  //af_print(masks);
  //auto iota = af::iota(4, u32);
  //af::array rows = iota(!(af::anyTrue(assignments, 1)));
  //af_print(rows);

  //while(rows.dims(0) > 0) {
    //af_print(rows);
    //rowLines(rows) = true;
    //af::array cols = iota(masks(rows, af::span) == 1);
    //if (cols.dims(0) == 0) {
      //break;
    //}
    //colLines(cols) = true;
    //af_print(cols);
    //rows = iota(assignments(af::span, cols) == 1);
    //af_print(rows);
  //}
  //rowLines = !rowLines;
  //af_print(rowLines);
  //af_print(colLines);

//}
