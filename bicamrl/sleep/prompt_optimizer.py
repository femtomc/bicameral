"""Prompt optimization for better AI interactions."""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..core.memory import Memory
from .roles import CommandRole

logger = logging.getLogger(__name__)

@dataclass
class PromptTemplate:
    """Reusable prompt template."""
    name: str
    description: str
    template: str
    variables: List[str]
    examples: List[Dict[str, Any]]
    success_rate: float = 0.0
    usage_count: int = 0
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing variable in template {self.name}: {e}")
            return self.template

@dataclass
class OptimizedPrompt:
    """Result of prompt optimization."""
    original: str
    optimized: str
    context_additions: List[Dict[str, Any]]
    templates_used: List[str]
    confidence: float
    reasoning: str

class PromptOptimizer:
    """Optimizes prompts based on knowledge base and patterns."""
    
    def __init__(self, memory: Memory):
        self.memory = memory
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()
        
    def _load_default_templates(self):
        """Load default prompt templates."""
        self.templates['code_generation'] = PromptTemplate(
            name='code_generation',
            description='Template for code generation tasks',
            template="""Generate {language} code for the following task:

Task: {task_description}

Requirements:
{requirements}

Context:
- Files involved: {files}
- Patterns to follow: {patterns}
- Constraints: {constraints}

Please provide:
1. Complete implementation
2. Brief explanation of approach
3. Any assumptions made""",
            variables=['language', 'task_description', 'requirements', 'files', 'patterns', 'constraints'],
            examples=[]
        )
        
        self.templates['bug_fix'] = PromptTemplate(
            name='bug_fix',
            description='Template for bug fixing tasks',
            template="""Debug and fix the following issue:

Error: {error_description}
File: {file_path}
Line: {line_number}

Error context:
{error_context}

Recent changes:
{recent_changes}

Similar past fixes:
{similar_fixes}

Provide:
1. Root cause analysis
2. Fix implementation
3. Prevention strategy""",
            variables=['error_description', 'file_path', 'line_number', 'error_context', 'recent_changes', 'similar_fixes'],
            examples=[]
        )
        
        self.templates['refactoring'] = PromptTemplate(
            name='refactoring',
            description='Template for refactoring tasks',
            template="""Refactor the following code:

Current implementation:
{current_code}

Refactoring goals:
{goals}

Project patterns:
{patterns}

Constraints:
{constraints}

Provide:
1. Refactored code
2. Explanation of changes
3. Impact assessment""",
            variables=['current_code', 'goals', 'patterns', 'constraints'],
            examples=[]
        )
        
    async def optimize_prompt(
        self,
        prompt: str,
        task_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        active_role: Optional[CommandRole] = None
    ) -> OptimizedPrompt:
        """Optimize a prompt based on patterns and context."""
        # Analyze the prompt
        analysis = await self._analyze_prompt(prompt, context)
        
        # Get relevant patterns and preferences
        patterns = await self._get_relevant_patterns(analysis)
        preferences = await self._get_relevant_preferences(analysis)
        
        # Select appropriate template
        template = self._select_template(analysis, task_type)
        
        # Build optimized prompt
        if template:
            optimized = await self._apply_template(template, prompt, analysis, patterns, preferences)
        else:
            optimized = await self._enhance_prompt(prompt, patterns, preferences)
        
        # Apply role-based enhancements if active
        if active_role:
            optimized = self._apply_role_enhancements(optimized, active_role)
            
        # Add relevant context
        context_additions = await self._select_context(analysis, patterns)
        
        # Add role-specific context if active
        if active_role:
            context_additions = self._add_role_context(context_additions, active_role)
        
        return OptimizedPrompt(
            original=prompt,
            optimized=optimized,
            context_additions=context_additions,
            templates_used=[template.name] if template else [],
            confidence=analysis.get('confidence', 0.7),
            reasoning=self._generate_reasoning(analysis, patterns, template, active_role)
        )
        
    async def _analyze_prompt(self, prompt: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze prompt to understand intent and requirements."""
        # Simple keyword-based analysis for now
        analysis = {
            'intent': self._detect_intent(prompt),
            'entities': self._extract_entities(prompt),
            'complexity': self._assess_complexity(prompt),
            'confidence': 0.8
        }
        
        # Check if it matches recent patterns
        recent_interactions = await self.memory.get_recent_context(limit=20)
        similar = self._find_similar_interactions(prompt, recent_interactions)
        
        if similar:
            analysis['similar_past'] = similar
            analysis['confidence'] = 0.9
            
        return analysis
        
    def _detect_intent(self, prompt: str) -> str:
        """Detect the intent of the prompt."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['fix', 'bug', 'error', 'issue']):
            return 'bug_fix'
        elif any(word in prompt_lower for word in ['generate', 'create', 'implement', 'write']):
            return 'code_generation'
        elif any(word in prompt_lower for word in ['refactor', 'improve', 'optimize', 'clean']):
            return 'refactoring'
        elif any(word in prompt_lower for word in ['explain', 'understand', 'what', 'how']):
            return 'explanation'
        elif any(word in prompt_lower for word in ['test', 'verify', 'check']):
            return 'testing'
        else:
            return 'general'
            
    def _extract_entities(self, prompt: str) -> Dict[str, List[str]]:
        """Extract entities from the prompt."""
        entities = {
            'files': [],
            'functions': [],
            'variables': [],
            'technologies': []
        }
        
        # Simple pattern matching for files
        import re
        
        # Match file paths
        file_pattern = r'[\w/]+\.\w+'
        entities['files'] = re.findall(file_pattern, prompt)
        
        # Match function names (camelCase or snake_case)
        func_pattern = r'\b[a-z][a-zA-Z0-9_]*(?:\(\))?'
        potential_funcs = re.findall(func_pattern, prompt)
        entities['functions'] = [f.rstrip('()') for f in potential_funcs if len(f) > 3]
        
        return entities
        
    def _assess_complexity(self, prompt: str) -> str:
        """Assess the complexity of the task."""
        word_count = len(prompt.split())
        
        if word_count < 20:
            return 'simple'
        elif word_count < 50:
            return 'moderate'
        else:
            return 'complex'
            
    def _find_similar_interactions(
        self,
        prompt: str,
        recent_interactions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find similar past interactions."""
        similar = []
        prompt_words = set(prompt.lower().split())
        
        for action in recent_interactions.get('recent_actions', []):
            if 'action' in action and isinstance(action['action'], str):
                action_words = set(action['action'].lower().split())
                similarity = len(prompt_words & action_words) / max(len(prompt_words), 1)
                
                if similarity > 0.3:
                    similar.append({
                        'action': action,
                        'similarity': similarity
                    })
                    
        return sorted(similar, key=lambda x: x['similarity'], reverse=True)[:3]
        
    async def _get_relevant_patterns(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get patterns relevant to the prompt."""
        all_patterns = await self.memory.get_all_patterns()
        relevant = []
        
        intent = analysis.get('intent', 'general')
        entities = analysis.get('entities', {})
        
        for pattern in all_patterns:
            # Check if pattern matches intent
            if pattern.get('pattern_type') == intent:
                relevant.append(pattern)
                continue
                
            # Check if pattern involves similar files
            pattern_desc = pattern.get('description', '').lower()
            for file in entities.get('files', []):
                if file.lower() in pattern_desc:
                    relevant.append(pattern)
                    break
                    
        return relevant[:5]  # Top 5 most relevant
        
    async def _get_relevant_preferences(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get preferences relevant to the task."""
        all_prefs = await self.memory.get_preferences()
        intent = analysis.get('intent', 'general')
        
        # Filter preferences by relevance
        relevant_prefs = {}
        
        if intent == 'code_generation':
            relevant_prefs['style'] = all_prefs.get('style', {})
            relevant_prefs['conventions'] = all_prefs.get('conventions', {})
        elif intent == 'testing':
            relevant_prefs['testing'] = all_prefs.get('testing', {})
        
        return relevant_prefs
        
    def _select_template(
        self,
        analysis: Dict[str, Any],
        task_type: Optional[str]
    ) -> Optional[PromptTemplate]:
        """Select the best template for the task."""
        intent = task_type or analysis.get('intent', 'general')
        
        if intent in self.templates:
            return self.templates[intent]
            
        # Find best matching template
        for name, template in self.templates.items():
            if intent in template.description.lower():
                return template
                
        return None
        
    async def _apply_template(
        self,
        template: PromptTemplate,
        original_prompt: str,
        analysis: Dict[str, Any],
        patterns: List[Dict[str, Any]],
        preferences: Dict[str, Any]
    ) -> str:
        """Apply a template to optimize the prompt."""
        # Extract variables from original prompt and context
        variables = {}
        
        # Map analysis to template variables
        if template.name == 'code_generation':
            variables['language'] = self._detect_language(original_prompt, preferences)
            variables['task_description'] = original_prompt
            variables['requirements'] = self._format_requirements(preferences)
            variables['files'] = ', '.join(analysis['entities'].get('files', ['relevant files']))
            variables['patterns'] = self._format_patterns(patterns)
            variables['constraints'] = self._format_constraints(preferences)
            
        elif template.name == 'bug_fix':
            variables['error_description'] = original_prompt
            files = analysis['entities'].get('files', [])
            variables['file_path'] = files[0] if files else 'unknown'
            variables['line_number'] = 'see context'
            variables['error_context'] = 'see attached context'
            variables['recent_changes'] = await self._get_recent_changes()
            variables['similar_fixes'] = self._format_similar_fixes(patterns)
            
        elif template.name == 'refactoring':
            variables['current_code'] = 'see context'
            variables['goals'] = original_prompt
            variables['patterns'] = self._format_patterns(patterns)
            variables['constraints'] = self._format_constraints(preferences)
            
        return template.format(**variables)
        
    async def _enhance_prompt(
        self,
        prompt: str,
        patterns: List[Dict[str, Any]],
        preferences: Dict[str, Any]
    ) -> str:
        """Enhance a prompt without using a template."""
        enhanced = prompt
        
        # Add pattern context if relevant
        if patterns:
            pattern_context = "\n\nRelevant patterns from this project:\n"
            for p in patterns[:3]:
                pattern_context += f"- {p['description']}\n"
            enhanced += pattern_context
            
        # Add preference context
        if preferences:
            pref_context = "\n\nProject preferences:\n"
            for category, prefs in preferences.items():
                for key, value in prefs.items():
                    pref_context += f"- {key}: {value}\n"
            enhanced += pref_context
            
        return enhanced
        
    async def _select_context(
        self,
        analysis: Dict[str, Any],
        patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Select relevant context to include."""
        context_items = []
        
        # Add files mentioned in the prompt
        for file in analysis['entities'].get('files', []):
            context_items.append({
                'type': 'file',
                'path': file,
                'reason': 'mentioned in prompt'
            })
            
        # Add files from relevant patterns
        for pattern in patterns[:2]:
            if 'sequence' in pattern:
                for item in pattern['sequence']:
                    if '.' in item:  # Likely a file
                        context_items.append({
                            'type': 'file',
                            'path': item,
                            'reason': f"part of pattern: {pattern['name']}"
                        })
                        
        # Get frequently accessed files
        recent_context = await self.memory.get_recent_context()
        for file_info in recent_context.get('top_files', [])[:3]:
            context_items.append({
                'type': 'file',
                'path': file_info['file'],
                'reason': f"frequently accessed ({file_info['count']} times)"
            })
            
        # Deduplicate
        seen_paths = set()
        unique_context = []
        for item in context_items:
            if item['path'] not in seen_paths:
                seen_paths.add(item['path'])
                unique_context.append(item)
                
        return unique_context[:5]  # Limit to 5 items
        
    def _generate_reasoning(
        self,
        analysis: Dict[str, Any],
        patterns: List[Dict[str, Any]],
        template: Optional[PromptTemplate],
        active_role: Optional[CommandRole] = None
    ) -> str:
        """Generate reasoning for the optimization."""
        reasoning_parts = []
        
        reasoning_parts.append(f"Detected intent: {analysis.get('intent', 'general')}")
        
        if patterns:
            reasoning_parts.append(f"Found {len(patterns)} relevant patterns")
            
        if template:
            reasoning_parts.append(f"Applied '{template.name}' template")
            
        if analysis.get('similar_past'):
            reasoning_parts.append("Found similar past interactions")
        
        if active_role:
            reasoning_parts.append(f"Applied '{active_role.name}' role context")
            
        return ". ".join(reasoning_parts)
        
    def _detect_language(self, prompt: str, preferences: Dict[str, Any]) -> str:
        """Detect programming language from prompt and preferences."""
        # Check preferences first
        if 'language' in preferences.get('general', {}):
            return preferences['general']['language']
            
        # Simple keyword detection
        languages = {
            'python': ['python', 'py', 'django', 'flask'],
            'javascript': ['javascript', 'js', 'node', 'react', 'vue'],
            'typescript': ['typescript', 'ts', 'angular'],
            'java': ['java', 'spring', 'maven'],
            'go': ['go', 'golang'],
            'rust': ['rust', 'cargo']
        }
        
        prompt_lower = prompt.lower()
        for lang, keywords in languages.items():
            if any(kw in prompt_lower for kw in keywords):
                return lang
                
        return 'python'  # Default
        
    def _format_requirements(self, preferences: Dict[str, Any]) -> str:
        """Format requirements from preferences."""
        requirements = []
        
        if 'style' in preferences:
            for key, value in preferences['style'].items():
                requirements.append(f"- {key}: {value}")
                
        return '\n'.join(requirements) if requirements else "- Follow project conventions"
        
    def _format_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """Format patterns for inclusion in prompt."""
        if not patterns:
            return "- Follow established project patterns"
            
        formatted = []
        for p in patterns[:3]:
            formatted.append(f"- {p['name']}: {p['description']}")
            
        return '\n'.join(formatted)
        
    def _format_constraints(self, preferences: Dict[str, Any]) -> str:
        """Format constraints from preferences."""
        constraints = []
        
        if 'rules' in preferences:
            for rule in preferences['rules'].values():
                constraints.append(f"- {rule}")
                
        return '\n'.join(constraints) if constraints else "- No specific constraints"
        
    async def _get_recent_changes(self) -> str:
        """Get description of recent changes."""
        recent = await self.memory.get_recent_context(limit=5)
        changes = []
        
        for action in recent.get('recent_actions', []):
            if action['action'] == 'edit_file':
                changes.append(f"- Modified {action.get('file', 'file')}")
                
        return '\n'.join(changes) if changes else "No recent changes"
        
    def _format_similar_fixes(self, patterns: List[Dict[str, Any]]) -> str:
        """Format similar fixes from patterns."""
        fixes = []
        
        for p in patterns:
            if p.get('pattern_type') == 'error' or 'fix' in p.get('name', '').lower():
                fixes.append(f"- {p['description']}")
                
        return '\n'.join(fixes) if fixes else "No similar fixes found"
        
    async def learn_from_interaction(
        self,
        prompt: str,
        optimized_prompt: str,
        outcome: Dict[str, Any]
    ):
        """Learn from the outcome of an optimized prompt."""
        # Record success/failure
        success = outcome.get('success', True)
        
        # Update template success rates if used
        if outcome.get('template_used'):
            template = self.templates.get(outcome['template_used'])
            if template:
                template.usage_count += 1
                # Simple moving average
                template.success_rate = (
                    (template.success_rate * (template.usage_count - 1) + (1.0 if success else 0.0)) /
                    template.usage_count
                )
                
        # Store successful optimizations as patterns
        if success and optimized_prompt != prompt:
            await self.memory.store.add_pattern({
                'name': 'Prompt optimization',
                'description': f"Optimized: {prompt[:50]}...",
                'pattern_type': 'prompt_optimization',
                'sequence': [prompt, optimized_prompt],
                'confidence': 0.8,
                'metadata': outcome
            })
    
    def _apply_role_enhancements(self, prompt: str, role: CommandRole) -> str:
        """Apply role-specific enhancements to the prompt."""
        enhanced = prompt
        
        # Add role context header
        role_header = f"\n[Operating in {role.name} mode]\n"
        
        # Add decision rules as context
        if role.decision_rules:
            rules_context = "\nApplicable guidelines:\n"
            for rule in sorted(role.decision_rules, key=lambda r: -r.priority)[:3]:
                rules_context += f"- {rule.to_prompt_instruction()}\n"
            enhanced = role_header + enhanced + rules_context
        else:
            enhanced = role_header + enhanced
        
        # Apply communication style modifiers
        style_notes = role.communication_profile.to_prompt_modifiers()
        if style_notes:
            enhanced += "\n\nCommunication style:\n"
            for note in style_notes[:2]:  # Limit to avoid verbosity
                enhanced += f"- {note}\n"
        
        return enhanced
    
    def _add_role_context(self, context_additions: List[Dict[str, Any]], role: CommandRole) -> List[Dict[str, Any]]:
        """Add role-specific context items."""
        role_context = context_additions.copy()
        
        # Add successful patterns from the role
        if role.successful_patterns:
            for pattern in role.successful_patterns[:2]:
                role_context.append({
                    'type': 'pattern',
                    'content': pattern,
                    'reason': f"successful pattern for {role.name} role"
                })
        
        # Add tool preferences as context hint
        if role.tool_preferences:
            top_tools = sorted(role.tool_preferences.items(), key=lambda x: -x[1])[:3]
            role_context.append({
                'type': 'tool_preference',
                'tools': [t[0] for t in top_tools],
                'reason': f"preferred tools for {role.name} role"
            })
        
        return role_context