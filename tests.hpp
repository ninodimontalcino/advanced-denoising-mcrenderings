#ifndef TESTS_H
#define TESTS_H

#include "flt.hpp"

#define TEST_FILES "tests/test_data/scene_Coateddiffuse"

#define TEST_F 3
#define TEST_R 10
#define TEST_KC 1.0
#define TEST_KF 2.0
#define TEST_TAU 0.001

// Right now this is a random value, used to compare if 2 floats are equal or not
#define FLOAT_TOLERANCE 0.00001

bool test_flt();
void load_buffer_from_txt(buffer *buf, std::string filename);
void free_buffer(buffer *buf);
void load_channel_from_txt(channel *buf, std::string filename);
void free_channel(channel *buf);

bool compare_buffers(buffer buf1, buffer buf2);
bool compare_scalar(scalar x, scalar y);

#endif // TESTS_H