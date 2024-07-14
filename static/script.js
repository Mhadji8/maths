function appendToDisplay(value) {
    document.getElementById('calc-display').value += value;
}

function clearDisplay() {
    document.getElementById('calc-display').value = '';
}

function calculate() {
    let expression = document.getElementById('calc-display').value;
    try {
        let result = eval(expression.replace('÷', '/').replace('x', '*'));
        document.getElementById('calc-display').value = result;
    } catch (e) {
        document.getElementById('calc-display').value = 'Erreur';
    }
}
