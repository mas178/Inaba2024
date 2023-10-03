JULIA_CMD = julia --project=../Inaba2024

.PHONY: all test test_simulation test_output test_entry_point format profiling update

all: update format test profiling

run:
	@echo "\nRunning src/EntryPoint.jl\n"
	nohup $(JULIA_CMD) --threads 8 src/EntryPoint.jl > log/run_all.log &

test: test_simulation test_output test_entry_point

test_simulation:
	@echo "\nRunning test/SimulationTest.jl\n"
	$(JULIA_CMD) test/SimulationTest.jl

test_output:
	@echo "\nRunning test/OutputTest.jl\n"
	$(JULIA_CMD) test/OutputTest.jl

test_entry_point:
	@echo "\nRunning test/EntryPointTest.jl\n"
	$(JULIA_CMD) test/EntryPointTest.jl

format:
	$(JULIA_CMD) -e 'using JuliaFormatter; format("src/")'
	$(JULIA_CMD) -e 'using JuliaFormatter; format("test/")'

profiling:
	$(JULIA_CMD) -i test/PerformanceProfiling.jl

update:
	julia -e "using Pkg; Pkg.update()"
	$(JULIA_CMD) -e "using Pkg; Pkg.update()"
