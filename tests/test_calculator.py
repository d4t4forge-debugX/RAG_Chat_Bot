from langgraph_backend import calculator


# verifies the calculator tool correctly adds two numbers
def test_add():
    result = calculator.invoke({"first_num": 2, "second_num": 3, "operation": "add"})
    assert result["result"] == 5


# verifies the calculator tool correctly subtracts two numbers
def test_sub():
    result = calculator.invoke({"first_num": 10, "second_num": 4, "operation": "sub"})
    assert result["result"] == 6


# verifies the calculator tool correctly multiplies two numbers
def test_mul():
    result = calculator.invoke({"first_num": 3, "second_num": 5, "operation": "mul"})
    assert result["result"] == 15


# verifies the calculator tool correctly divides two numbers
def test_div():
    result = calculator.invoke({"first_num": 10, "second_num": 2, "operation": "div"})
    assert result["result"] == 5


# verifies dividing by zero returns an error dict instead of raising or crashing
def test_div_by_zero():
    result = calculator.invoke({"first_num": 10, "second_num": 0, "operation": "div"})
    assert "error" in result


# verifies an unrecognized operation name returns an error dict instead of silently doing nothing
def test_unsupported_operation():
    result = calculator.invoke({"first_num": 1, "second_num": 1, "operation": "mod"})
    assert "error" in result