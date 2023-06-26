ALL_COMMANDS = ['Line', 'Arc', 'Circle', 'EOS', 'SOL', 'Ext']
LINE_IDX = ALL_COMMANDS.index('Line')
ARC_IDX = ALL_COMMANDS.index('Arc')
CIRCLE_IDX = ALL_COMMANDS.index('Circle')
EOS_IDX = ALL_COMMANDS.index('EOS')
SOL_IDX = ALL_COMMANDS.index('SOL')
EXT_IDX = ALL_COMMANDS.index('Ext')

N_ARGS_SKETCH = 5 # sketch parameters: x, y, alpha, f, r
N_ARGS_PLANE = 3 # sketch plane orientation: theta, phi, gamma
N_ARGS_TRANS = 4 # sketch plane origin + sketch bbox size: p_x, p_y, p_z, s
N_ARGS_EXT_PARAM = 4 # extrusion parameters: e1, e2, b, u
N_ARGS_EXT = N_ARGS_PLANE + N_ARGS_TRANS + N_ARGS_EXT_PARAM
N_ARGS = N_ARGS_SKETCH + N_ARGS_EXT  # 16

QUANTUM_BIT = 8
ARGS_DIM = 2**QUANTUM_BIT  # 256
VOCAB_SIZE = len(ALL_COMMANDS) + ARGS_DIM * N_ARGS  # 6 + 256x16 = 4102

PAD_IDX = 4102