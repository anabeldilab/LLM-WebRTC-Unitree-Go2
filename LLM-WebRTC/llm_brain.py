import json
import logging
import requests
import pydantic

logger = logging.getLogger("LLMGo2Brain")
logger.setLevel(logging.INFO)
logger.propagate = True

ALLOWED_ACTIONS = {
    "DAMP": "El robot apaga sus motores y colapsa de forma insegura. Solo para emergencias, no lo uses a menos que quieras que el robot se desplome sin control.",
    "BALANCE_STAND": "Activa el control de equilibrio dinámico. Solo se puede usar mientras está de pie. Se usa para iniciar movimientos dinámicos como 'MOVE', 'TROT_RUN' o 'FREE_WALK'.",
    "STOP_MOVE": "Detiene inmediatamente cualquier movimiento, manteniendo la postura actual. Para el movimiento iniciado. No apaga el control de equilibrio, por lo que el robot seguirá manteniendo la postura de pie sin caerse.",
    "STAND_UP": "El robot se levanta de una posición tumbada o sentada a la postura de equilibrio estático.",
    "STAND_DOWN": "El robot baja su centro de gravedad de forma lenta y segura hasta quedar tumbado en el suelo con las patas recogidas.",
    "RECOVERY_STAND": "Maniobra para levantar al robot después de una fuerte caída.",
    "SIT": "Postura estándar de sentado (patas traseras flexionadas, patas delanteras extendidas).",
    "HELLO": "Levanta la pata delantera hacia adelante y hacia atrás. Se usa como gesto de saludo o para llamar la atención.",
    "STRETCH": "El robot estira su torso y extremidades hacia adelante y hacia atrás.",
    "HEART": "El robot hace el gesto de un corazón con sus patas delanteras. Úsalo para expresar afecto o amor.",
    "SCRAPE": "El robot imita rascar el aire frente a él alternando las patas delanteras mientras soporta el peso en las patas traseras.",
    "DANCE1": "Baile rítmico y coordinado con pasos laterales.",
    "DANCE2": "Secuencia de baile de cuerpo entero más compleja y dinámica.",
    "TROT_RUN": "Marcha de trote dinámico. Bueno para un desplazamiento rápido.",
    "FREE_WALK": "Marcha de caminata suave y estable. Bueno para movimientos lentos o exploración.",
    "FRONT_JUMP": "Salta hacia adelante. ¡Ten cuidado con lo que hay en frente!"
}

class LLMAction(pydantic.BaseModel):
    type: str  # "action" or "move"
    value: str = None  # For type="action", the action name (e.g., "SIT", "HEART")
    params: dict = None  # For type="move", the parameters (x, y, yaw, duration)

class LLMResponse(pydantic.BaseModel):
    actions: list[LLMAction]
    description: str

class LLMGo2Brain:
    def __init__(self, local_model="llama3:latest", host="localhost", port=11434):
        self.ollama_url = f"http://{host}:{port}/api/generate"
        self.model = local_model
        
        actions_desc = "\n".join([f"- {k}: {v}" for k, v in ALLOWED_ACTIONS.items()])
        self.system_prompt = f"""
        Eres Jonay, el cerebro de un robot Unitree Go2.
        Convierte las instrucciones en comandos JSON técnicos.
        
        ACCIONES DISPONIBLES:
        {actions_desc}

        EL COMANDO 'MOVE' es para movimiento continuo en una dirección durante una duración específica. Parámetros:
        - x: adelante(1)/atrás(-1).
        - y: izquierda(1)/derecha(-1).
        - yaw: girar izquierda(1)/girar derecha(-1).
        - duration: tiempo en segundos (MÍNIMO 1, NUNCA 0).

        REGLAS DE FORMATO INQUEBRANTABLES:
        1. La clave "type" SOLO puede ser "action" o "move". ¡NUNCA pongas el nombre de la acción ahí!
        2. Para type="action", el nombre de la acción (SIT, HEART) va en la clave "value".
        3. Para type="move", los parámetros (x, y, yaw, duration) van en un diccionario dentro de la clave "params".
        4. El JSON debe tener exactamente las claves "actions" y "description".
        5. Si la instrucción no es clara o viola las reglas, responde con una lista vacía de acciones y una descripción explicando el problema en formato json. (ej: {{"actions": [], "description": "La instrucción no está clara."}})

        EJEMPLOS DE COMPORTAMIENTO ESPERADO:
        Usuario: "Jonay, ¿me quieres?"
        {{
            "actions": [{{"type": "action", "value": "HEART"}}],
            "description": "El usuario expresa afecto, respondiendo con HEART."
        }}

        Usuario: "Jonay, da un paso hacia adelante durante dos segundos"
        {{
            "actions": [{{"type": "move", "params": {{"x": 0.3, "y": 0, "yaw": 0, "duration": 2}}}}],
            "description": "El usuario pide moverse hacia adelante, usando move con x=0.3."
        }}
        """

    def process(self, text):
        logger.info(f"🧠 LLM (Jonay) Thinking how to execute: '{text}'...")
        try:
            payload = {
                "model": self.model,
                "prompt": text,
                "system": self.system_prompt,
                "stream": False,
                "format": "json",
                "temperature": 0.2,  # Low temperature for more deterministic output
            }
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            response.raise_for_status()
            
            content = response.json().get("response", "{}")
            content = content.replace("```json", "").replace("```", "").strip()
            
            try:
                validated_data = LLMResponse.model_validate_json(content)
                
                logger.info(f"💡 Reasoning: {validated_data.description}")
                
                actions_dict = [action.model_dump(exclude_none=True) for action in validated_data.actions]
                
                if not actions_dict:
                    logger.warning(f"⚠️ Jonay decided not to execute any physical action.")
                else:
                    logger.info(f"📋 Generated plan:\n{json.dumps(actions_dict, indent=2)}")
                
                return actions_dict, validated_data.description
                    
            except pydantic.ValidationError as ve:
                logger.error(f"❌ Invalid LLM format. The model violated the rules:\n{ve}")
                logger.debug(f"Raw content: {content}")
                return None, "Structural Error (Pydantic Validation Failed)"
                
        except requests.exceptions.Timeout:
            logger.error("❌ Timeout: Ollama took too long to respond.")
            return None, "Timeout Error"
        except Exception as e:
            logger.error(f"❌ Critical error in Brain: {e}")
            return None, f"Critical Error: {e}"