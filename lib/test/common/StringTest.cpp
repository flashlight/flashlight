/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <string>

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/lib/common/String.h"

using namespace fl::lib;

TEST(StringTest, StringTrim) {
  EXPECT_EQ(trim(""), "");
  EXPECT_EQ(trim("     "), "");
  EXPECT_EQ(trim(" \n \tabc   "), "abc");
  EXPECT_EQ(trim("ab     ."), "ab     .");
  EXPECT_EQ(trim("|   ab cd   "), "|   ab cd");
}

TEST(StringTest, ReplaceAll) {
  std::string in = "\tSomewhere, something incredible is waiting to be known.";
  replaceAll(in, "\t", "   ");
  EXPECT_EQ(in, "   Somewhere, something incredible is waiting to be known.");
  replaceAll(in, "   ", "");
  EXPECT_EQ(in, "Somewhere, something incredible is waiting to be known.");
  replaceAll(in, "some", "any");
  EXPECT_EQ(in, "Somewhere, anything incredible is waiting to be known.");
  replaceAll(in, " ", "");
  EXPECT_EQ(in, "Somewhere,anythingincredibleiswaitingtobeknown.");
}

TEST(StringTest, StringSplit) {
  using Pieces = std::vector<std::string>;
  const std::string& input = " ;abc; de;;";
  // char delimiters
  EXPECT_EQ(split(';', input), (Pieces{" ", "abc", " de", "", ""}));
  EXPECT_EQ(split(';', input, true), (Pieces{" ", "abc", " de"}));
  EXPECT_EQ(split('X', input), (Pieces{input}));
  // string delimiters
  EXPECT_EQ(split(";;", input), (Pieces{" ;abc; de", ""}));
  EXPECT_EQ(split(";;", input, true), (Pieces{" ;abc; de"}));
  EXPECT_EQ(split("ac", input), (Pieces{input}));
  EXPECT_THROW(split("", input), std::invalid_argument);
  // multi-char delimiters
  EXPECT_EQ(splitOnAnyOf("bce", input), (Pieces{" ;a", "", "; d", ";;"}));
  EXPECT_EQ(splitOnAnyOf("bce", input, true), (Pieces{" ;a", "; d", ";;"}));
  EXPECT_EQ(splitOnAnyOf("", input), (Pieces{input}));
  // whitespace
  EXPECT_EQ(splitOnWhitespace(input), (Pieces{"", ";abc;", "de;;"}));
  EXPECT_EQ(splitOnWhitespace(input, true), (Pieces{";abc;", "de;;"}));
}

TEST(StringTest, StringJoin) {
  using Pieces = std::vector<std::string>;
  // from vector
  EXPECT_EQ(join("", Pieces{"a", "b", "", "c"}), "abc");
  EXPECT_EQ(join(",", Pieces{"a", "b", "", "c"}), "a,b,,c");
  EXPECT_EQ(join(",\n", Pieces{"a", "b", "", "c"}), "a,\nb,\n,\nc");
  EXPECT_EQ(join("abc", Pieces{}), "");
  EXPECT_EQ(join("abc", Pieces{"abc"}), "abc");
  EXPECT_EQ(join("abc", Pieces{"abc", "abc"}), "abcabcabc");
  // from iterator range
  Pieces input{"in", "te", "re", "st", "ing"};
  EXPECT_EQ(join("", input.begin(), input.end()), "interesting");
  EXPECT_EQ(join("", input.begin(), input.end() - 1), "interest");
  EXPECT_EQ(join("", input.begin(), input.begin()), "");
  EXPECT_EQ(join("e", input.begin() + 1, input.end() - 1), "teereest");
}

TEST(StringTest, StringFormat) {
  EXPECT_EQ(format("a%sa", "bbb"), "abbba");
  EXPECT_EQ(format("%%%c%s%c", 'a', "bbb", 'c'), "%abbbc");
  EXPECT_EQ(format("0x%08x", 0x0023ffaa), "0x0023ffaa");
  EXPECT_EQ(format("%5s", "abc"), "  abc");
  EXPECT_EQ(format("%.3f", 3.1415926), "3.142");

  std::string big(2000, 'a');
  EXPECT_EQ(format("(%s)", big.c_str()), std::string() + "(" + big + ")");
}

TEST(StringTest, Uniq) {
  std::vector<int> uq1 = {5, 6, 6, 8, 9, 8, 8, 8};
  dedup(uq1);
  ASSERT_THAT(uq1, ::testing::ElementsAre(5, 6, 8, 9, 8));

  std::vector<int> uq2 = {1, 1, 1, 1, 1};
  dedup(uq2);
  ASSERT_THAT(uq2, ::testing::ElementsAre(1));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

// Resolve directory for sample dictionary
#ifdef FL_DICTIONARY_TEST_DIR
  loadPath = FL_DICTIONARY_TEST_DIR;
#endif

  return RUN_ALL_TESTS();
}
