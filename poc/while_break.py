def test_nested_while_break():
    while True:
        while True:
            i = 0
            break
        if i == 0:
            print("i == 0")
            break
        print("i != 0")
        break
    print(f"{i=}")


if __name__ == "__main__":
    test_nested_while_break()
