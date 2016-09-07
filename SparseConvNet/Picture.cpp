#include "Picture.h"

#include <iostream>

std::string Picture::identify() { return std::string(); }
Picture::Picture(int label) : label(label) {}
Picture::~Picture() {
    //std::cout << "Picture destroy" << std::endl;
}
