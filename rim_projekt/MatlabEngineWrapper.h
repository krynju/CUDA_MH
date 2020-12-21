#pragma once

#include <memory>
#include <string>
#include <MatlabEngine.hpp>

class MatlabEngineWrapper
{
public:
	static auto& instance()
	{
		static MatlabEngineWrapper inst;
		return inst;
	}

	void setArray(const std::string & name, const float * x, size_t N);

private:
	MatlabEngineWrapper();

	auto _createArray(const float * x, size_t N);
	
	auto _setArray(const std::string & name, const matlab::data::Array & arr);

	std::unique_ptr<matlab::engine::MATLABEngine> engine_;
};

