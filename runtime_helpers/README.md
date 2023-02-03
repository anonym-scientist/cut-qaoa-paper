# qiskit-runtime-helpers

A collection of functions that simplify the creation of Qiskit Runtime programs and the interaction with them.

## Usage

Preparations:
```Python
IBMQ.load_account()
provider = IBMQ.get_provider(hub='your-hub', group='your-group', project='your-project')
metadata = {'name': 'program-name', 'max_execution_time': 28800, 'description': 'A test program'}
program_options = {'backend_name': 'ibmq_qasm_simulator'}
inputs = {'your': 'input'}
```

Build a runtime program that includes all your local imports:

```Python
program = build_program('path/your_runtime_program.py', python_paths=['project-directory'])
```

Upload the program:

```Python
program_id = upload_program(provider, data=program, meta_data=metadata)
```

Start the program:
```Python
job = start_program(provider, program_id, program_options, inputs)
```
