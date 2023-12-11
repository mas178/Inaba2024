JULIA_CMD = julia --project=../Inaba2024

.PHONY: all test test_simulation test_output test_entry_point format profiling update

all: update format test profiling

run:
	@echo "\nRunning src/EntryPoint.jl\n"
	nohup $(JULIA_CMD) --threads 8 src/EntryPoint.jl > log/run_all.log &

test: test_simulation test_output test_entry_point test_sim_plot

test_simulation:
	@echo "\nRunning test/Ar1Test.jl\n"
	$(JULIA_CMD) test/Ar1Test.jl
	@echo "\nRunning test/ModelPopulationTest.jl\n"
	$(JULIA_CMD) test/ModelPopulationTest.jl
	@echo "\nRunning test/ModelPayoffTest.jl\n"
	$(JULIA_CMD) test/ModelPayoffTest.jl

test_output:
	@echo "\nRunning test/OutputTest.jl\n"
	$(JULIA_CMD) test/OutputTest.jl

test_entry_point:
	@echo "\nRunning test/EntryPointTest.jl\n"
	$(JULIA_CMD) test/EntryPointTest.jl

test_sim_plot:
	@echo "\nRunning test/SimPlotTest.jl\n"
	$(JULIA_CMD) test/SimPlotTest.jl

format:
	$(JULIA_CMD) -e 'using JuliaFormatter; format("src/")'
	$(JULIA_CMD) -e 'using JuliaFormatter; format("test/")'

profiling:
	$(JULIA_CMD) -i test/PerformanceProfiling.jl

update:
	julia -e "using Pkg; Pkg.update()"
	$(JULIA_CMD) -e "using Pkg; Pkg.update()"
