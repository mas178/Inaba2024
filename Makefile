JULIA_CMD = julia --project=../Inaba2024

.PHONY: test test_simulation test_output test_entry_point format profiling

test: test_simulation test_output test_entry_point

test_simulation:
	@echo "\nRunning test/SimulationTest.jl\n"
	$(JULIA_CMD) test/SimulationTest.jl

test_output:
	@echo "\nRunning test/OutputTest.jl\n"
	$(JULIA_CMD) test/OutputTest.jl

test_entry_point:
	@echo "\nRunning test/EntryPoint.jl\n"
	$(JULIA_CMD) test/EntryPointTest.jl

format:
	$(JULIA_CMD) -e 'using JuliaFormatter; format("src/Simulation.jl")'
	$(JULIA_CMD) -e 'using JuliaFormatter; format("src/Output.jl")'
	$(JULIA_CMD) -e 'using JuliaFormatter; format("test/")'

profiling:
	$(JULIA_CMD) -i test/PerformanceProfiling.jl
