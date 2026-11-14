"""Clean outcome checks for worked-reference corpus curation, never assigned labels."""
import ast
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from fractions import Fraction
from functools import lru_cache
import io
import itertools
import json
import re
import signal
from types import SimpleNamespace


def extract_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else None


@lru_cache(maxsize=4)
def latin_grids(n):
    if n != 4:
        raise ValueError('Only audited 4x4 Futoshiki is supported')
    permutations = tuple(itertools.permutations(range(1, n + 1)))
    grids = []
    def extend(rows):
        if len(rows) == n:
            grids.append(tuple(rows))
            return
        for row in permutations:
            if all(all(row[j] != other[j] for j in range(n)) for other in rows):
                extend(rows + [row])
    extend([])
    assert len(grids) == 576
    return tuple(grids)


def futoshiki_spec(game):
    n = game['metadata']['grid_size']
    fixed = []
    for line in game['question'].splitlines():
        fields = line.strip().split()
        if len(fields) == n and all(t == 'X' or t.isdigit() and 1 <= int(t) <= n for t in fields):
            fixed.append(tuple(None if t == 'X' else int(t) for t in fields))
    constraints = tuple((int(a)-1, int(b)-1, int(c)-1, int(d)-1, op)
                        for a, b, op, c, d in re.findall(
                            r'\((\d+),\s*(\d+)\)\s*([<>])\s*\((\d+),\s*(\d+)\)', game['question']))
    assert len(fixed) == n
    assert set(constraints) == {(*a, *b, op) for a, b, op in game['metadata']['constraints']}
    return n, tuple(fixed), constraints


def valid_futoshiki(grid, spec):
    n, fixed, constraints = spec
    try:
        return (len(grid) == n and all(len(row) == n for row in grid)
                and all(set(row) == set(range(1, n+1)) for row in grid)
                and all({grid[i][j] for i in range(n)} == set(range(1, n+1)) for j in range(n))
                and all(fixed[i][j] is None or fixed[i][j] == grid[i][j] for i in range(n) for j in range(n))
                and all((grid[a][b] < grid[c][d]) if op == '<' else (grid[a][b] > grid[c][d])
                        for a, b, c, d, op in constraints))
    except (TypeError, ValueError, IndexError):
        return False


@lru_cache(maxsize=1024)
def futoshiki_solutions(spec):
    return tuple(grid for grid in latin_grids(spec[0]) if valid_futoshiki(grid, spec))


def dyck_error(game):
    stack, pairs = [], {')': '(', ']': '[', '}': '{', '>': '<'}
    text = game['metadata']['bracket_string']
    for index, char in enumerate(text, 1):
        if char in pairs.values():
            stack.append(char)
        else:
            assert char in pairs
            if not stack or stack.pop() != pairs[char]:
                return index
    return len(text)+1 if stack else -1


def mathador_correct(game, answer, verifier):
    expression = verifier.extract_answer(answer)
    if not expression or len(expression) > 512:
        return False
    metadata = game['metadata']
    if Counter(re.findall(r'\d+', expression)) != Counter(str(n) for n in metadata['numbers']):
        return False
    if any(c not in '0123456789() \n\t' and c not in metadata['operators'] for c in expression):
        return False
    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
        raise ValueError('Not a permitted arithmetic expression')
    try:
        return abs(evaluate(ast.parse(expression, mode='eval').body) - Fraction(str(metadata['result']))) < Fraction(1, 10**10)
    except (ValueError, SyntaxError, ZeroDivisionError, RecursionError):
        return False


class VerifierTimeout(TimeoutError):
    pass


def verify_response(row, text, verifier_classes, timeout=10):
    """Return tri-state correctness and independent native reward/coverage."""
    game = json.loads(row['extra_info']['original_game_data_str'])
    task = row['data_source']
    answer = extract_answer(text)
    if answer is None:
        return {'correct': False, 'reward': 0., 'verified': True, 'failure': None,
                'native_correct': False, 'native_verified': True, 'checker': 'native_required_answer_tag'}
    verifier = verifier_classes[task]()
    data = SimpleNamespace(**game)
    if task == 'skyscraper_puzzle':
        verifier.n = game['metadata']['n']
    result = {'correct': None, 'reward': None, 'verified': False, 'failure': None,
              'native_correct': None, 'native_verified': False, 'checker': 'native_clean_verifier'}
    old_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    assert previous_timer[0] == 0, 'Cannot nest verifier deadlines'
    def deadline(*_):
        raise VerifierTimeout('Corpus verifier deadline exceeded')
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        output = io.StringIO()
        try:
            with redirect_stdout(output), redirect_stderr(output):
                native = bool(verifier.verify(data, answer))
            swallowed_error = re.search(r'(?im)^.*(?:验证时出错|验证出错|Traceback|Error verifying|验证过程中.*错误).*$', output.getvalue())
            if swallowed_error:
                result['native_failure'] = swallowed_error[0][:300]
            else:
                result.update(native_correct=native, native_verified=True)
        except Exception as error:
            result['native_failure'] = type(error).__name__ + ': ' + str(error)[:300]
        if task == 'futoshiki':
            spec = futoshiki_spec(game)
            solutions = futoshiki_solutions(spec)
            result['checker'] = 'stated_givens_and_constraints_exhaustive_576_latin_squares'
            result['puzzle_satisfiable'] = bool(solutions)
            if solutions:
                grid = verifier.extract_answer(answer)
                correct = grid is not None and valid_futoshiki(grid, spec)
            else:
                correct = bool(re.fullmatch(r'(?is)\s*(?:no (?:valid )?solution(?: exists)?|unsatisfiable|impossible|无解|没有解)[.!。]?\s*', answer))
        elif task == 'dyck_language_errors':
            result['checker'] = 'independent_stack_first_error_position'
            correct = bool(re.fullmatch(r'-?\d+', answer)) and int(answer) == dyck_error(game)
        elif task == 'mathador':
            result['checker'] = 'independent_fraction_ast_original_numbers_operators_target'
            correct = mathador_correct(game, answer, verifier)
        else:
            if not result['native_verified']:
                raise ValueError(result.get('native_failure', 'Native verifier coverage missing'))
            correct = result['native_correct']
        result.update(correct=bool(correct), reward=float(bool(correct)), verified=True)
    except Exception as error:
        result.update(failure=type(error).__name__ + ': ' + str(error)[:300])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
    return result
