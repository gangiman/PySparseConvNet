#pragma once
#include "Picture.h"
#include "Rng.h"
#include <vector>
#include <string>
#include "types.h"
#include "SpatiallySparseDataset.h"
#include <glob.h>

#include <cassert>
#include <list>
#include <map>

// Class providing fixed-size (by number of records)
// LRU-replacement cache of a function with signature
// V f(K).
// MAP should be one of std::map or std::unordered_map.
// Variadic template args used to deal with the
// different type argument signatures of those
// containers; the default comparator/hash/allocator
// will be used.
template <typename K, typename V> class lru_cache
{
public:

    typedef K key_type;
    typedef V value_type;

    // Key access history, most recent at back
    using key_tracker_type = std::list<key_type>;

    // Key to value and key history iterator
    using key_to_value_type = std::map<key_type, std::pair<value_type, typename std::list<key_type>::iterator>>;

    // Constuctor specifies the cached function and
    // the maximum number of records to be stored
    lru_cache(value_type (*f)(const key_type&), size_t c)
        : _fn(f)
        ,_capacity(c)
    {
        assert(_capacity!=0);
    }

    // Obtain value of the cached function for k
    value_type operator[](const key_type& k) {

        // Attempt to find existing record
        const auto it = _key_to_value.find(k);

        if (it == _key_to_value.end()) {
            // We don't have it:
            // Evaluate function and create new record
            const value_type v = _fn(k);
            insert(k,v);

            // Return the freshly computed value
            return v;
        } else {
            // We do have it:
            // Update access record by moving
            // accessed key to back of list
            _key_tracker.splice(_key_tracker.end(), _key_tracker, (*it).second.second);

            // Return the retrieved value
            return (*it).second.first;
        }
    }

    // Obtain the cached keys, most recently used element
    // at head, least recently used at tail.
    // This method is provided purely to support testing.
    template <typename IT> void get_keys(IT dst) const {
        typename key_tracker_type::const_reverse_iterator src =_key_tracker.rbegin();
        while (src!=_key_tracker.rend()) {
            *dst++ = *src++;
        }
    }

private:

    // Record a fresh key-value pair in the cache
    void insert(const key_type& k,const value_type& v) {

        // Method is only called on cache misses
        assert(_key_to_value.find(k) == _key_to_value.end());

        // Make space if necessary
        if (_key_to_value.size() == _capacity)
            evict();

        // Record k as most-recently-used key
        typename key_tracker_type::iterator it =_key_tracker.insert(_key_tracker.end(),k);

        // Create the key-value entry,
        // linked to the usage record.
        _key_to_value.insert(std::make_pair(k, std::make_pair(v,it)));
        // No need to check return,
        // given previous assert.
    }

    // Purge the least-recently-used element in the cache
    void evict() {

        // Assert method is never called when cache is empty
        assert(!_key_tracker.empty());

        // Identify least recently used key
        const auto it = _key_to_value.find(_key_tracker.front());
        assert(it!=_key_to_value.end());

        // Erase both elements to completely purge record
        _key_to_value.erase(it);
        _key_tracker.pop_front();
    }

    // The function to be cached
    value_type (*_fn)(const key_type&);

    // Maximum number of key-value pairs to be retained
    const size_t _capacity;

    // Key access history
    key_tracker_type _key_tracker;

    // Key-to-value lookup
    key_to_value_type _key_to_value;
};


std::shared_ptr<Picture> get_fucking_picture(const int& i);

class SpatiallySparseDataset {
  RNG rng;

public:
  std::string name;
  std::string header;
  //std::vector<Picture*> pictures;
  using super_cache = lru_cache<int, std::shared_ptr<Picture>>;
  super_cache real_pictures;// = super_cache(&get_picture, 10000);
  std::vector<int> pictures;

    SpatiallySparseDataset() : real_pictures(&get_fucking_picture, size_t(10000)) {

    }

  int renderSize;
  int nFeatures;
  int nClasses;
  batchType type;
  void summary();
  void shuffle();
  SpatiallySparseDataset extractValidationSet(float p = 0.1);
  void subsetOfClasses(std::vector<int> activeClasses);
  SpatiallySparseDataset subset(int n);
  SpatiallySparseDataset balancedSubset(int n);
  void repeatSamples(int reps); // Make dataset seem n times bigger (i.e. for
                                // small datasets to avoid having v. small
                                // training epochs)
};

std::vector<std::string> globVector(const std::string &pattern);
