# Qualitative Evaluation Methods

Qualitative methods focus on **human judgment and contextual understanding** rather than numerical scores. These methods are widely used to evaluate machine learning models, AI systems, and user-facing applications.

---

## 1. Human Evaluation

### Overview
Human evaluation involves experts or end users manually assessing model outputs based on predefined criteria such as accuracy, relevance, coherence, or usefulness.

### Common Criteria
- Correctness
- Clarity
- Fluency
- Relevance
- Ethical alignment

### Example
In a chatbot system, human reviewers read generated responses and rate them on a scale (e.g., 1–5) for helpfulness and correctness.

### Advantages
- High-quality, nuanced feedback
- Captures subjective and contextual errors

### Limitations
- Time-consuming
- Expensive
- Can be biased or inconsistent

---

## 2. Crowdsourcing Feedback

### Overview
Crowdsourcing uses a large pool of non-expert users (via platforms like Amazon Mechanical Turk or internal user panels) to collect feedback at scale.

### Common Tasks
- Rating responses
- Comparing outputs (A/B testing)
- Labeling sentiment or intent

### Example
Multiple crowd workers compare two recommendation systems and choose which one provides better suggestions.

### Advantages
- Scalable and cost-effective
- Faster data collection
- Diverse perspectives

### Limitations
- Lower expertise level
- Quality control required
- Potential noisy or inconsistent labels

---

## 3. Scenario Testing

### Overview
Scenario testing evaluates system behavior in **predefined real-world situations** to ensure robustness, safety, and reliability.

### Key Aspects
- Edge cases
- Failure scenarios
- Ethical or safety-critical situations

### Example
Testing an autonomous driving model under scenarios such as poor weather, sudden obstacles, or pedestrian crossings.

### Advantages
- Reveals real-world weaknesses
- Improves reliability and safety
- Helps validate deployment readiness

### Limitations
- Requires careful scenario design
- May not cover all real-world variations

---

## Comparison Summary

| Method              | Who Evaluates | Scale      | Best For                         |
|---------------------|--------------|------------|----------------------------------|
| Human Evaluation    | Experts      | Small      | Deep qualitative insights        |
| Crowdsourcing       | General users| Large      | Fast, scalable feedback          |
| Scenario Testing    | Testers      | Medium     | Robustness & real-world behavior |

---

## Conclusion

Qualitative evaluation methods complement quantitative metrics by providing **human-centered insights**. Combining human evaluation, crowdsourcing feedback, and scenario testing leads to more reliable and user-aligned systems.
