## Refined Tasks List

1. **Relocate core Tetris components from `tetris-ai-models` to `gym-simpletetris`:**

   - Move `BASIC_ACTIONS` and `ACTION_COMBINATIONS` to appropriate files in `gym_simpletetris`. (IN PROGRESS)
   - Transfer `simplify_board` function to the Tetris engine. (IN PROGRESS)
   - Move game-specific constants to the Tetris environment configuration.
   - Relocate reward calculation from `tetris-ai-models/utils/reward_functions.py` to `gym_simpletetris/tetris/tetris_engine.py`.
   - ~~**MOVED** `helpful_utils.py` to `gym-simpletetris`.~~ (DONE)
   - ~~Dependency injection for scoring system~~ (DONE)

2. **Implement new logging system for `gym-simpletetris`:**

   - Design a flexible logging system for the Tetris environment.
   - Implement basic logging functionality (game state, actions, rewards).
   - Add configuration options for log levels and output formats.
   - Ensure the logger can be easily extended for future needs.

3. **Refactor existing logger in `tetris-ai-models`:**

   - Review and update the existing logger to focus solely on AI training metrics.
   - Remove any Tetris-specific logging that's now handled by `gym-simpletetris`.
   - Ensure compatibility with the new `gym-simpletetris` logger.

4. **Initial integration and testing:**

   - Integrate all relocated components and the new logger into `gym-simpletetris`.
   - Create basic tests for relocated components and the new logger.
   - Update `tetris-ai-models` to use the new `gym-simpletetris` package and its logger.

5. **Baseline performance benchmarking:**

   - Set up a benchmarking framework for `gym-simpletetris`.
   - Run comprehensive performance tests on the current implementation.
   - Use both loggers to capture relevant metrics for analysis.

6. **optimise relocated components:**

   - Prioritise optimisation based on benchmark results.
   - Focus on reward calculation logic, using PyTorch/NumPy optimisations.
   - Optimise other components as needed.
   - Use the `gym-simpletetris` logger to track performance improvements.

7. **Comprehensive testing and validation:**

   - Expand the test suite for the refactored and optimised `gym-simpletetris`.
   - Validate correctness of optimised components and logging systems.
   - Ensure no regression in functionality.

8. **Post-optimisation benchmarking:**

   - Re-run performance benchmarks on the optimised code.
   - Use both loggers to compare results with the baseline.
   - Identify any remaining performance bottlenecks.

9. **Documentation and code quality:**

   - Update documentation for both `gym-simpletetris` and `tetris-ai-models`.
   - Document the new logging systems and their integration.
   - Emphasise code quality, readability, and adherence to standards.

10. **Dependency management:**

    - Review and update dependencies in both packages.
    - Ensure compatibility between the packages and their loggers.

11. **Final updates to `tetris-ai-models`:**

    - Remove any remaining relocated code.
    - Finalize the integration with the new `gym-simpletetris` and its logger.
    - Ensure `tetris-ai-models` remains focused on AI model implementation and training.

12. **Update README.md files:**

    - Update READMEs for both packages to reflect new structure and logging systems.
    - Provide examples of how to use and configure the new loggers.

13. **Final review and testing:**
    - Conduct a final code review of both packages, focusing on logger integration.
    - Run a full suite of tests to ensure everything works as expected.
    - Verify that logging outputs are correct and useful for both packages.
