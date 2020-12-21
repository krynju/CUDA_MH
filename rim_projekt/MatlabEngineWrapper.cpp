#include "MatlabEngineWrapper.h"

MatlabEngineWrapper::MatlabEngineWrapper()
	: engine_(matlab::engine::connectMATLAB())
{
}

auto MatlabEngineWrapper::_createArray(const float* x, size_t N)
{
	matlab::data::ArrayFactory factory;

	auto array = factory.createArray<double>({ N });

	std::copy(x, x + N, array.begin());

	return array;
}

auto MatlabEngineWrapper::_setArray(const std::string& name, const matlab::data::Array& arr)
{
	engine_->setVariable(name, arr);
}

void MatlabEngineWrapper::setArray(const std::string & name, const float* x, size_t N)
{
	try
	{
		_setArray(name, _createArray(x, N));
	}
	catch (const matlab::Exception& e)
	{
		std::cout << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << __FILE__ << __LINE__ << "Unknown exception.\n";
	}
}
