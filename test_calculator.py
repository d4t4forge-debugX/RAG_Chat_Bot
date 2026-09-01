from langgraph_backend import calculator


def test_add():
    result = calculator.invoke({"first_num": 2, "second_num": 3, "operation": "add"})
    assert result["result"] == 5


def test_sub():
    result = calculator.invoke({"first_num": 10, "second_num": 4, "operation": "sub"})
    assert result["result"] == 6


def test_mul():
    result = calculator.invoke({"first_num": 3, "second_num": 5, "operation": "mul"})
    assert result["result"] == 15


def test_div():
    result = calculator.invoke({"first_num": 10, "second_num": 2, "operation": "div"})
    assert result["result"] == 5


def test_div_by_zero():
    result = calculator.invoke({"first_num": 10, "second_num": 0, "operation": "div"})
    assert "error" in result


def test_unsupported_operation():
    result = calculator.invoke({"first_num": 1, "second_num": 1, "operation": "mod"})
    assert "error" in result