#ifndef _STOPWATCH_HXX_
#define _STOPWATCH_HXX_

#include <chrono>

template <typename real>
class stopwatch
{
public:
	void reset();
	real elapsed();

private:
	using clock = std::chrono::high_resolution_clock;
	clock::time_point t0 { clock::now() };
};


template <typename real> inline
void stopwatch<real>::reset()
{
	t0 = clock::now();
}

template <typename real> inline
real stopwatch<real>::elapsed()
{
	using seconds = std::chrono::duration<real,std::ratio<1,1>>;

	auto tnow = clock::now();
	auto duration = std::chrono::duration_cast<seconds>( tnow - t0 );

	return duration.count();
}



#endif

