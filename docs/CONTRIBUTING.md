# Contributing to PlotSenseAI Demo Projects 🤝

Thank you for your interest in contributing to the PlotSenseAI Hackathon Demo Projects! This guide will help you get started with contributing to our open-source educational resources.

## 🌟 Ways to Contribute

We welcome all types of contributions:

- 🐛 **Bug Reports**: Found something that doesn't work? Let us know!
- ✨ **Feature Requests**: Have ideas for improvements? Share them!
- 📝 **Documentation**: Help improve our guides, tutorials, and examples
- 🧪 **Tests**: Add test cases to improve code reliability
- 💻 **Code**: Fix bugs, add features, or optimize performance
- 🎨 **Examples**: Create new demo projects or enhance existing ones
- 🌍 **Translations**: Help make our content accessible globally

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/PlotSenseAI-Hackathon-Demo-Projects.git
cd PlotSenseAI-Hackathon-Demo-Projects

# Add upstream remote
git remote add upstream https://github.com/Havilah-Blockchain-Studios/PlotSenseAI-Hackathon-Demo-Projects.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv dev-env
source dev-env/bin/activate  # On Windows: dev-env\Scripts\activate

# Install dependencies
pip install plotsense pandas numpy matplotlib jupyter streamlit pytest

# Install development dependencies
pip install black flake8 pre-commit
```

### 3. Create a Feature Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

## 📋 Contribution Guidelines

### Code Style

We follow Python best practices and PEP 8 guidelines:

```bash
# Format code with Black
black .

# Check code style with flake8
flake8 .

# Run before committing
pre-commit run --all-files
```

**Key Style Points**:
- Use descriptive variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Use type hints where appropriate
- Follow existing code patterns in each project

### Documentation Standards

**For Code**:
```python
def zscore_anomaly_detection(data, threshold=1.5):
    """
    Detect anomalies using Z-score method.

    Args:
        data (list or pd.Series): Numeric data for analysis
        threshold (float): Z-score cutoff for anomaly detection

    Returns:
        pd.DataFrame: DataFrame with anomaly detection results

    Example:
        >>> data = [1, 2, 3, 100, 4, 5]
        >>> result = zscore_anomaly_detection(data, threshold=2.0)
        >>> print(result[result['anomaly'] == True])
    """
```

**For Markdown**:
- Use clear headings and structure
- Include code examples with syntax highlighting
- Add emoji for visual appeal (sparingly)
- Provide context and explanations
- Include troubleshooting sections

### Testing Requirements

**For New Features**:
```python
# tests/test_new_feature.py
import pytest
from plotsense_anomaly import new_feature

def test_new_feature_basic():
    """Test basic functionality"""
    result = new_feature([1, 2, 3])
    assert len(result) == 3

def test_new_feature_edge_cases():
    """Test edge cases"""
    # Empty data
    with pytest.raises(ValueError):
        new_feature([])

    # Single value
    result = new_feature([5])
    assert len(result) == 1
```

**Run Tests**:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_detection.py

# Run with coverage
pytest --cov=plotsense_anomaly
```

## 🎯 Types of Contributions

### 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment Information**:
   - Operating system and version
   - Python version
   - PlotSenseAI version
   - Other relevant package versions

2. **Reproduction Steps**:
   ```
   1. Navigate to project_two
   2. Run `python examples/demo_anomaly_detection.py`
   3. Error occurs at line X
   ```

3. **Expected vs Actual Behavior**:
   - What you expected to happen
   - What actually happened
   - Error messages or screenshots

4. **Additional Context**:
   - Sample data (if relevant)
   - Configuration files
   - Logs or output

### ✨ Feature Requests

For feature requests, please provide:

1. **Problem Description**: What problem does this solve?
2. **Proposed Solution**: How would you like it to work?
3. **Alternatives Considered**: What other approaches did you consider?
4. **Use Cases**: How would this benefit users?
5. **Implementation Ideas**: Any thoughts on how to implement?

### 💻 Code Contributions

#### Small Changes
For small bug fixes or minor improvements:
1. Create an issue first (unless it's truly trivial)
2. Fork and create a branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

#### Large Changes
For major features or significant changes:
1. **Start with an issue**: Discuss the approach first
2. **Design document**: For complex features, create a design doc
3. **Incremental development**: Break into smaller, reviewable chunks
4. **Coordinate**: Work with maintainers to avoid conflicts

### 📝 Documentation Contributions

**Types of Documentation**:
- **Tutorials**: Step-by-step guides for beginners
- **How-to Guides**: Solutions for specific problems
- **API Reference**: Technical documentation
- **Examples**: Practical use cases and code samples

**Best Practices**:
- Start with user needs
- Use clear, simple language
- Include working code examples
- Test all code snippets
- Add screenshots for UI-related content

## 🔄 Pull Request Process

### 1. Pre-Submission Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated (if applicable)
- [ ] CHANGELOG updated (for significant changes)
- [ ] Branch is up-to-date with upstream main

### 2. Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Test improvement

## Testing
- [ ] Tests pass locally
- [ ] New tests added (if applicable)
- [ ] Manual testing completed

## Screenshots
(If applicable)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Breaking changes documented
```

### 3. Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and style checks
2. **Code Review**: Maintainers review code quality and design
3. **Discussion**: Address feedback and questions
4. **Approval**: At least one maintainer approval required
5. **Merge**: Maintainer merges after all checks pass

## 🏷️ Issue and PR Labels

**Type Labels**:
- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to docs
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed

**Priority Labels**:
- `priority: high`: Critical issues
- `priority: medium`: Important but not urgent
- `priority: low`: Nice to have

**Status Labels**:
- `status: in progress`: Currently being worked on
- `status: needs review`: Ready for review
- `status: blocked`: Cannot proceed

## 🎓 Learning Resources

### For New Contributors
- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)
- [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- [Open Source Guide](https://opensource.guide/how-to-contribute/)

### For PlotSenseAI Development
- [PlotSenseAI Documentation](https://docs.plotsense.ai)
- [Python Packaging Guide](https://packaging.python.org/)
- [Jupyter Notebook Best Practices](https://jupyter-notebook.readthedocs.io/)

### For Data Science
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

## 🌟 Recognition

We believe in recognizing our contributors:

- **Contributors List**: All contributors listed in README
- **Release Notes**: Major contributors mentioned in releases
- **Hall of Fame**: Outstanding contributors featured on website
- **Swag**: Stickers and swag for regular contributors
- **References**: LinkedIn recommendations for significant contributors

## 🤔 Questions and Support

### Before Contributing
1. **Search existing issues**: Your question might already be answered
2. **Read documentation**: Check our guides and tutorials
3. **Join discussions**: Participate in GitHub Discussions

### Getting Help
- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: support@havilahacademy.org for sensitive matters
- **Discord**: Join our community server (link in README)

### Mentorship Program
New contributors can request mentorship:
- Pair with experienced contributors
- Guidance on first contributions
- Code review and feedback
- Career advice in open source

## 📜 Code of Conduct

### Our Pledge
We pledge to make participation in our project a harassment-free experience for everyone, regardless of:
- Age, body size, disability, ethnicity
- Gender identity and expression
- Level of experience, education, socio-economic status
- Nationality, personal appearance, race, religion
- Sexual identity and orientation

### Our Standards
**Positive behavior includes**:
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what's best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes**:
- Trolling, insulting, or derogatory comments
- Public or private harassment
- Publishing private information without consent
- Conduct that could reasonably be considered inappropriate

### Enforcement
Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at support@havilahacademy.org. All complaints will be reviewed and investigated promptly and fairly.

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

## 🎉 Thank You!

Every contribution, no matter how small, helps make this project better for everyone. We appreciate your time and effort in helping build better educational resources for the PlotSenseAI community!

---

**Happy Contributing! 🚀**

*Questions? Feel free to reach out in our [GitHub Discussions](https://github.com/HavilahAcademy/PlotSenseAI-Hackathon-Demo-Projects/discussions) or contact us at support@havilahacademy.org*