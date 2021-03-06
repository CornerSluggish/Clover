/**
 *  Clover: Quantized 4-bit Linear Algebra Library
 *              ______ __
 *             / ____// /____  _   __ ___   _____
 *            / /    / // __ \| | / // _ \ / ___/
 *           / /___ / // /_/ /| |/ //  __// /
 *           \____//_/ \____/ |___/ \___//_/
 *
 *  Copyright 2018 Alen Stojanov       (astojanov@inf.ethz.ch)
 *                 Tyler Michael Smith (tyler.smith@inf.ethz.ch)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <cinttypes>
#include <iostream>

using namespace std;

void test_endianness ()
{
    volatile uint32_t i=0x01234567;
    int endiness = (*((uint8_t*)(&i))) == 0x67;

    cout << "======================================================================" << endl;
    if (endiness) {
        cout << "Endianness: Little Endian" << endl;
    } else {
        cout << "Endianness: Big Endian" << endl;
        cout << endl;
        cout << "This code is not designed for Big Endian. Exiting ... ";
        exit(1);
    }
    cout << "======================================================================" << endl;
    cout << endl;
}

