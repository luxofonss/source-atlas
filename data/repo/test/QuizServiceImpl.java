package com.example.demo.service;

import com.example.demo.model.User;
import com.example.demo.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.jms.annotation.JmsListener;
import org.springframework.jms.core.JmsTemplate;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;

@RestController
@RequestMapping("/api/users")
@Service
public class UserManagementService {

    @Autowired
    private KafkaTemplate<String, String> kafkaTemplate;
    
    @Autowired
    private JmsTemplate jmsTemplate;
    
    @Autowired
    private ExternalApiService externalApiService;
    
    @Autowired
    private UserRepository userRepository;

    // REST ENDPOINTS
    @GetMapping
    public ResponseEntity<List<User>> getAllUsers() {
        List<User> users = userRepository.findAll();
        return ResponseEntity.ok(users);
    }

    @PostMapping
    public ResponseEntity<User> createUser(@RequestBody User user) {
        User savedUser = userRepository.save(user);
        
        // Send notification via Kafka
        sendKafkaMessage("user-events", "User created: " + savedUser.getId());
        
        // Send message to JMS queue
        sendJmsMessage("user.queue", "New user registered: " + savedUser.getEmail());
        
        return ResponseEntity.ok(savedUser);
    }

    @GetMapping("/{id}")
    public ResponseEntity<User> getUserById(@PathVariable Long id) {
        // Call to another class method
        UserDetails details = externalApiService.fetchUserDetails(id);
        
        User user = userRepository.findById(id).orElse(null);
        if (user != null) {
            user.setLastAccessTime(LocalDateTime.now());
            userRepository.save(user);
        }
        
        return ResponseEntity.ok(user);
    }

    // SPRING BOOT SCHEDULED JOB
    @Scheduled(fixedRate = 300000) // Run every 5 minutes
    public void cleanupExpiredSessions() {
        System.out.println("Running cleanup job at: " + LocalDateTime.now());
        
        // Simulate cleanup logic
        List<User> inactiveUsers = userRepository.findInactiveUsers();
        for (User user : inactiveUsers) {
            sendKafkaMessage("cleanup-events", "Cleaning up user: " + user.getId());
        }
        
        System.out.println("Cleanup completed for " + inactiveUsers.size() + " users");
    }

    // KAFKA PRODUCER
    public void sendKafkaMessage(String topic, String message) {
        kafkaTemplate.send(topic, message);
        System.out.println("Kafka message sent to topic '" + topic + "': " + message);
    }

    // KAFKA LISTENER
    @KafkaListener(topics = "user-events", groupId = "user-service-group")
    public void handleUserEvent(String message) {
        System.out.println("Received Kafka message: " + message);
        
        // Process the event
        if (message.contains("User created")) {
            // Send welcome email logic here
            sendJmsMessage("email.queue", "Send welcome email for: " + message);
        }
    }

    // JMS MESSAGE QUEUE PRODUCER
    public void sendJmsMessage(String queue, String message) {
        jmsTemplate.convertAndSend(queue, message);
        System.out.println("JMS message sent to queue '" + queue + "': " + message);
    }

    // JMS MESSAGE QUEUE LISTENER
    @JmsListener(destination = "user.queue")
    public void handleUserNotification(String message) {
        System.out.println("Received JMS message: " + message);
        
        // Create audit log entry
        AuditLog auditLog = new AuditLog();
        auditLog.setMessage(message);
        auditLog.setTimestamp(LocalDateTime.now());
        auditLog.setProcessed(true);
        
        System.out.println("Audit log created: " + auditLog);
    }

    // NESTED CLASS
    public static class AuditLog {
        private String message;
        private LocalDateTime timestamp;
        private boolean processed;

        public AuditLog() {}

        public String getMessage() {
            return message;
        }

        public void setMessage(String message) {
            this.message = message;
        }

        public LocalDateTime getTimestamp() {
            return timestamp;
        }

        public void setTimestamp(LocalDateTime timestamp) {
            this.timestamp = timestamp;
        }

        public boolean isProcessed() {
            return processed;
        }

        public void setProcessed(boolean processed) {
            this.processed = processed;
        }

        @Override
        public String toString() {
            return "AuditLog{" +
                    "message='" + message + '\'' +
                    ", timestamp=" + timestamp +
                    ", processed=" + processed +
                    '}';
        }
    }

    // Inner class for user statistics
    public class UserStatistics {
        private int totalUsers;
        private int activeUsers;
        private LocalDateTime lastCalculated;

        public void calculateStats() {
            this.totalUsers = userRepository.findAll().size();
            this.activeUsers = userRepository.findActiveUsers().size();
            this.lastCalculated = LocalDateTime.now();
            
            // Send stats to Kafka
            String statsMessage = String.format("Stats: Total=%d, Active=%d", totalUsers, activeUsers);
            sendKafkaMessage("stats-topic", statsMessage);
        }

        // Getters and setters
        public int getTotalUsers() { return totalUsers; }
        public void setTotalUsers(int totalUsers) { this.totalUsers = totalUsers; }
        
        public int getActiveUsers() { return activeUsers; }
        public void setActiveUsers(int activeUsers) { this.activeUsers = activeUsers; }
        
        public LocalDateTime getLastCalculated() { return lastCalculated; }
        public void setLastCalculated(LocalDateTime lastCalculated) { this.lastCalculated = lastCalculated; }
    }
}

// SEPARATE CLASS THAT IMPLEMENTS AND EXTENDS
@Service
class ExternalApiService extends BaseApiService implements ApiCallable {
    
    @Override
    public UserDetails fetchUserDetails(Long userId) {
        // Simulate external API call
        System.out.println("Fetching user details for ID: " + userId);
        
        UserDetails details = new UserDetails();
        details.setUserId(userId);
        details.setProfileComplete(true);
        details.setLastLoginTime(LocalDateTime.now().minusDays(1));
        
        // Call parent class method
        5("fetchUserDetails", userId.toString());
        
        return details;
    }
    
    @Override
    public boolean validateUser(Long userId) {
        System.out.println("Validating user: " + userId);
        return userId != null && userId > 0;
    }
    
    // Additional method specific to this service
    public void syncUserData(Long userId) {
        if (validateUser(userId)) {
            UserDetails details = fetchUserDetails(userId);
            // Sync logic here
            System.out.println("User data synced for: " + userId);
        }
    }
}

// BASE CLASS TO EXTEND
abstract class BaseApiService {
    protected void logApiCall(String method, String parameter) {
        System.out.println("API Call logged: " + method + " with parameter: " + parameter);
    }
    
    protected void handleApiError(Exception e) {
        System.err.println("API Error occurred: " + e.getMessage());
    }
}

// INTERFACE TO IMPLEMENT
interface ApiCallable {
    UserDetails fetchUserDetails(Long userId);
    boolean validateUser(Long userId);
}

// SUPPORTING CLASSES
class UserDetails {
    private Long userId;
    private boolean profileComplete;
    private LocalDateTime lastLoginTime;

    // Getters and setters
    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }
    
    public boolean isProfileComplete() { return profileComplete; }
    public void setProfileComplete(boolean profileComplete) { this.profileComplete = profileComplete; }
    
    public LocalDateTime getLastLoginTime() { return lastLoginTime; }
    public void setLastLoginTime(LocalDateTime lastLoginTime) { this.lastLoginTime = lastLoginTime; }
}