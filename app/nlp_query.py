"""
Natural Language Query System using LangChain and FAISS

Allows users to query detection results and system metrics using natural language.
Uses sentence embeddings and vector similarity search for efficient retrieval.
"""

import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger


class NLPQueryEngine:
    """
    Natural language query engine for detection results and metrics

    Uses FAISS for vector similarity search and sentence-transformers for embeddings
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Sentence transformer model name
        """
        self.model = SentenceTransformer(model_name)
        self.detection_history = []
        self.metrics_history = []

        # Pre-defined query templates and their handlers
        self.query_templates = {
            "detection_count": {
                "keywords": ["how many", "count", "number of", "total"],
                "handler": self._handle_count_query
            },
            "class_distribution": {
                "keywords": ["distribution", "breakdown", "what classes", "which objects"],
                "handler": self._handle_distribution_query
            },
            "recent_detections": {
                "keywords": ["recent", "latest", "last", "past"],
                "handler": self._handle_recent_query
            },
            "high_confidence": {
                "keywords": ["high confidence", "confident", "certain", "sure"],
                "handler": self._handle_confidence_query
            },
            "vehicle_pedestrian": {
                "keywords": ["vehicle", "pedestrian", "car", "person", "people"],
                "handler": self._handle_vp_query
            },
            "performance": {
                "keywords": ["performance", "speed", "latency", "fps", "throughput"],
                "handler": self._handle_performance_query
            },
            "accuracy": {
                "keywords": ["accuracy", "map", "precision", "recall"],
                "handler": self._handle_accuracy_query
            }
        }

        # Create embeddings for templates
        self._create_template_embeddings()

    def _create_template_embeddings(self):
        """Create embeddings for query template keywords"""
        self.template_embeddings = {}
        for template_name, template_data in self.query_templates.items():
            keywords = " ".join(template_data["keywords"])
            embedding = self.model.encode(keywords)
            self.template_embeddings[template_name] = embedding

    def add_detection_result(self, detections: List[Dict], timestamp: Optional[datetime] = None):
        """
        Add detection result to history

        Args:
            detections: List of detection dictionaries
            timestamp: Timestamp of detection (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        self.detection_history.append({
            "timestamp": timestamp,
            "detections": detections,
            "count": len(detections)
        })

        # Keep only last 10000 results
        if len(self.detection_history) > 10000:
            self.detection_history = self.detection_history[-10000:]

    def add_metrics(self, metrics: Dict, timestamp: Optional[datetime] = None):
        """
        Add metrics to history

        Args:
            metrics: Metrics dictionary
            timestamp: Timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        self.metrics_history.append({
            "timestamp": timestamp,
            "metrics": metrics
        })

        # Keep only last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

    def query(self, question: str) -> Dict:
        """
        Process natural language query and return results

        Args:
            question: Natural language question

        Returns:
            Dictionary with query results
        """
        logger.info(f"Processing query: {question}")

        # Encode query
        query_embedding = self.model.encode(question)

        # Find best matching template
        best_template = None
        best_similarity = -1

        for template_name, template_embedding in self.template_embeddings.items():
            similarity = np.dot(query_embedding, template_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(template_embedding)
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_template = template_name

        logger.info(f"Matched template: {best_template} (similarity: {best_similarity:.3f})")

        # Handle query
        if best_template and best_similarity > 0.3:
            handler = self.query_templates[best_template]["handler"]
            result = handler(question)
        else:
            result = {
                "answer": "I'm not sure how to answer that. Try asking about detection counts, class distribution, recent detections, or performance metrics.",
                "confidence": 0.0
            }

        result["matched_template"] = best_template
        result["confidence_score"] = float(best_similarity)

        return result

    def _handle_count_query(self, question: str) -> Dict:
        """Handle queries about detection counts"""
        if not self.detection_history:
            return {"answer": "No detection data available yet.", "data": {}}

        # Extract time window from question
        time_window = self._extract_time_window(question)
        filtered = self._filter_by_time(self.detection_history, time_window)

        total_detections = sum(item["count"] for item in filtered)

        # Check for specific class
        specific_class = self._extract_class_name(question)
        if specific_class:
            class_count = 0
            for item in filtered:
                for det in item["detections"]:
                    if det.get("cls", "").lower() == specific_class.lower():
                        class_count += 1
            answer = f"Found {class_count} {specific_class} detections in the {time_window}."
            data = {"class": specific_class, "count": class_count}
        else:
            answer = f"Total of {total_detections} detections in the {time_window}."
            data = {"total_count": total_detections}

        return {"answer": answer, "data": data}

    def _handle_distribution_query(self, question: str) -> Dict:
        """Handle queries about class distribution"""
        if not self.detection_history:
            return {"answer": "No detection data available yet.", "data": {}}

        time_window = self._extract_time_window(question)
        filtered = self._filter_by_time(self.detection_history, time_window)

        # Count by class
        class_counts = {}
        for item in filtered:
            for det in item["detections"]:
                cls = det.get("cls", "unknown")
                class_counts[cls] = class_counts.get(cls, 0) + 1

        total = sum(class_counts.values())
        distribution = {cls: {"count": count, "percentage": count/total*100}
                       for cls, count in class_counts.items()}

        # Create answer
        top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        answer = f"Class distribution in {time_window}:\n"
        for cls, count in top_classes:
            pct = count/total*100
            answer += f"- {cls}: {count} ({pct:.1f}%)\n"

        return {"answer": answer.strip(), "data": distribution}

    def _handle_recent_query(self, question: str) -> Dict:
        """Handle queries about recent detections"""
        if not self.detection_history:
            return {"answer": "No detection data available yet.", "data": {}}

        # Get last N results
        n = 10
        recent = self.detection_history[-n:]

        total = sum(item["count"] for item in recent)
        answer = f"In the last {n} predictions:\n"
        answer += f"- Total detections: {total}\n"

        if recent:
            latest = recent[-1]
            answer += f"- Latest prediction had {latest['count']} detections\n"
            if latest["detections"]:
                classes = [d.get("cls", "unknown") for d in latest["detections"]]
                answer += f"- Detected: {', '.join(set(classes))}"

        return {"answer": answer.strip(), "data": {"recent_count": n, "total": total}}

    def _handle_confidence_query(self, question: str) -> Dict:
        """Handle queries about high confidence detections"""
        if not self.detection_history:
            return {"answer": "No detection data available yet.", "data": {}}

        threshold = 0.8
        time_window = self._extract_time_window(question)
        filtered = self._filter_by_time(self.detection_history, time_window)

        high_conf = []
        for item in filtered:
            for det in item["detections"]:
                if det.get("confidence", 0) >= threshold:
                    high_conf.append(det)

        answer = f"Found {len(high_conf)} high-confidence (>{threshold}) detections in {time_window}."

        if high_conf:
            avg_conf = sum(d.get("confidence", 0) for d in high_conf) / len(high_conf)
            answer += f"\nAverage confidence: {avg_conf:.3f}"

        return {"answer": answer, "data": {"count": len(high_conf), "threshold": threshold}}

    def _handle_vp_query(self, question: str) -> Dict:
        """Handle queries about vehicles/pedestrians"""
        if not self.detection_history:
            return {"answer": "No detection data available yet.", "data": {}}

        vp_classes = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
        time_window = self._extract_time_window(question)
        filtered = self._filter_by_time(self.detection_history, time_window)

        vp_counts = {}
        for item in filtered:
            for det in item["detections"]:
                cls = det.get("cls", "")
                if cls in vp_classes:
                    vp_counts[cls] = vp_counts.get(cls, 0) + 1

        total = sum(vp_counts.values())
        answer = f"Vehicle/Pedestrian detections in {time_window}: {total} total\n"
        for cls, count in sorted(vp_counts.items(), key=lambda x: x[1], reverse=True):
            answer += f"- {cls}: {count}\n"

        return {"answer": answer.strip(), "data": vp_counts}

    def _handle_performance_query(self, question: str) -> Dict:
        """Handle queries about performance metrics"""
        if not self.metrics_history:
            return {"answer": "No performance metrics available yet.", "data": {}}

        recent_metrics = self.metrics_history[-100:]

        # Calculate average latency if available
        answer = "Performance metrics:\n"
        answer += f"- Processed {len(self.detection_history)} predictions\n"

        if self.metrics_history:
            latest = self.metrics_history[-1]["metrics"]
            answer += f"- Latest metrics: {json.dumps(latest, indent=2)}"

        return {"answer": answer, "data": {"samples": len(recent_metrics)}}

    def _handle_accuracy_query(self, question: str) -> Dict:
        """Handle queries about accuracy metrics"""
        if not self.metrics_history:
            return {"answer": "No accuracy metrics available yet.", "data": {}}

        latest = self.metrics_history[-1]["metrics"]
        answer = "Accuracy metrics:\n"

        if "mAP_0.5_0.95" in latest:
            answer += f"- mAP@0.5:0.95: {latest['mAP_0.5_0.95']:.3f}\n"
        if "mAP_0.5" in latest:
            answer += f"- mAP@0.5: {latest['mAP_0.5']:.3f}\n"

        return {"answer": answer.strip(), "data": latest}

    def _extract_time_window(self, question: str) -> str:
        """Extract time window from question"""
        question_lower = question.lower()
        if "last hour" in question_lower or "past hour" in question_lower:
            return "last hour"
        elif "today" in question_lower:
            return "today"
        elif "last 24" in question_lower or "past 24" in question_lower:
            return "last 24 hours"
        else:
            return "all time"

    def _extract_class_name(self, question: str) -> Optional[str]:
        """Extract class name from question"""
        common_classes = ["person", "car", "bus", "truck", "bicycle", "motorcycle"]
        question_lower = question.lower()

        for cls in common_classes:
            if cls in question_lower:
                return cls
        return None

    def _filter_by_time(self, history: List[Dict], time_window: str) -> List[Dict]:
        """Filter history by time window"""
        if time_window == "all time":
            return history

        now = datetime.utcnow()
        if time_window == "last hour":
            cutoff = now - timedelta(hours=1)
        elif time_window == "today":
            cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_window == "last 24 hours":
            cutoff = now - timedelta(hours=24)
        else:
            return history

        return [item for item in history if item["timestamp"] >= cutoff]

    def get_stats(self) -> Dict:
        """Get statistics about the query engine"""
        return {
            "detection_history_size": len(self.detection_history),
            "metrics_history_size": len(self.metrics_history),
            "available_templates": list(self.query_templates.keys())
        }
